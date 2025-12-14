"""Cleaner utility for monthly snapshots.

Features:
- Detects encoding and delimiter heuristically
- Normalizes column names to canonical schema
- Cleans numeric fields and description
- Writes cleaned CSVs to `Proyecto/data/cleaned/cleaned_<origname>.csv` (UTF-8)
- Writes a JSON/text report per input in `Proyecto/ml/reports/`

Usage:
    python Proyecto/ml/cleaner.py --input <file>
    python Proyecto/ml/cleaner.py --run-all --input-dir ../data

"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import hashlib
from collections import Counter

# Try encodings in order
ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin1"]
COMMON_DELIMS = [",", ";", "\t", "|"]

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
CLEANED_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"

# Per-file overrides for known messy snapshots
OVERRIDES = {
    'cuadro-de-inventario-de-medicamentos-feb.-2024': {
        'header': None,
        'encoding': 'latin-1',
        'expected_cols': ['CODIGO','DESCRIPCION','PRECIO UNITARIO B/.',
                          'EXISTENCIA ALMACEN CDPA FEBRERO 2024',
                          'EXISTENCIA ALMACEN CDDI FEBRERO 2024',
                          'EXISTENCIA ALMACEN CDCH FEBRERO 2024',
                          'TOTAL DE EXISTENCIAS DISPONIBLES FEBRERO 2024',
                          'MONTO DE EXISTENCIAS EN B/.'],
        'malformed_check': False
    },
    'inventario-de-medicamentos-marzo-2024': {
        'header': None,
        # try several encodings and semicolon separator (legacy script used sep=';')
        'encodings': ['utf-8','latin-1','cp1252'],
        'sep': ';',
        'expected_cols': ['CODIGO','DESCRIPCION','PRECIO UNITARIO B/.',
                          'EXISTENCIA ALMACEN CDPA MARZO 2024',
                          'EXISTENCIA ALMACEN CDDI MARZO 2024',
                          'EXISTENCIA ALMACEN CDCH MARZO 2024',
                          'TOTAL DE EXISTENCIAS DISPONIBLES MARZO 2024',
                          'MONTO DE EXISTENCIAS EN B/.'],
        'malformed_check': True
    }
}


def detect_encoding_and_delim(path: Path, sample_bytes: int = 8192) -> Tuple[str, str]:
    """Return (encoding, delimiter) guessed from a file sample."""
    sample = None
    with open(path, "rb") as f:
        sample = f.read(sample_bytes)
    # try encodings
    for enc in ENCODINGS_TO_TRY:
        try:
            text = sample.decode(enc)
        except Exception:
            continue
        # try csv.Sniffer
        try:
            sn = csv.Sniffer()
            dialect = sn.sniff(text)
            delim = dialect.delimiter
        except Exception:
            # fallback heuristics
            delim = None
            for d in COMMON_DELIMS:
                if d in text:
                    delim = d
                    break
            if delim is None:
                delim = ","
        return enc, delim
    # fallback
    return "latin1", ","


def find_column(cols, candidates):
    """Return the first matching column name (case-insensitive) from candidates, or None."""
    lookup = {c.upper(): c for c in cols}
    for cand in candidates:
        if cand is None:
            continue
        if cand.upper() in lookup:
            return lookup[cand.upper()]
    return None


def clean_numeric_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).copy()
    s = s.str.replace('"', '', regex=False).str.replace("'", '', regex=False)
    s = s.str.replace('\u00A0', '', regex=False).str.strip()
    s = s.str.replace(',', '', regex=False)
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')


CANONICAL_COLS = [
    'CODIGO', 'DESCRIPCION_LIMPIA', 'DESCRIPCION_ORIG',
    'snapshot_date', 'precio_unitario',
    'existencia_cdpa', 'existencia_cddi', 'existencia_cdch', 'total_existencias', 'monto_existencias'
]


def normalize_dataframe(df: pd.DataFrame, source_name: Optional[str] = None) -> pd.DataFrame:
    # detect likely columns
    cols = list(df.columns)
    codigo_col = find_column(cols, ['CODIGO', 'CODE', 'ID', 'COD'])
    desc_col = find_column(cols, ['DESCRIPCION', 'DESCRIPCIÓN', 'MEDICAMENTO', 'NOMBRE', 'DESCRIPCION_ORIG'])
    price_col = find_column(cols, ['PRECIO UNITARIO B/.', 'PRECIO_UNITARIO', 'PRECIO', 'precio_unitario'])
    # snapshot/date
    snap_col = find_column(cols, ['snapshot_date', 'FECHA', 'FECHA_SNAPSHOT', 'FECHA_MOVIMIENTO'])

    df2 = df.copy()
    # rename detected to canonical temporary names
    rename_map = {}
    if codigo_col:
        rename_map[codigo_col] = 'CODIGO'
    if desc_col:
        rename_map[desc_col] = 'DESCRIPCION_ORIG'
    if price_col:
        rename_map[price_col] = 'PRECIO_RAW'
    if snap_col:
        rename_map[snap_col] = 'snapshot_date'

    if rename_map:
        df2 = df2.rename(columns=rename_map)

    # Ensure CODIGO exists
    if 'CODIGO' not in df2.columns:
        # try first column
        df2['CODIGO'] = df2.iloc[:, 0].astype(str)

    # Build DESCRIPCION_ORIG if missing
    if 'DESCRIPCION_ORIG' not in df2.columns:
        # choose first text-like column
        text_cols = [c for c in df2.columns if df2[c].dtype == object]
        if text_cols:
            df2['DESCRIPCION_ORIG'] = df2[text_cols[0]].astype(str)
        else:
            df2['DESCRIPCION_ORIG'] = df2['CODIGO'].astype(str)

    # DESCRIPCION_LIMPIA
    df2['DESCRIPCION_LIMPIA'] = df2['DESCRIPCION_ORIG'].astype(str).str.strip().str.upper()

    # precio_unitario
    if 'PRECIO_RAW' in df2.columns:
        df2['precio_unitario'] = clean_numeric_series(df2['PRECIO_RAW'])
    elif 'precio_unitario' in df2.columns:
        df2['precio_unitario'] = clean_numeric_series(df2['precio_unitario'])
    else:
        df2['precio_unitario'] = pd.NA

    # try to find existencia / monto columns loosely
    existencia_cols = [c for c in df2.columns if 'EXIST' in str(c).upper() or 'ALMAC' in str(c).upper() or 'TOTAL' in str(c).upper()]
    for c in existencia_cols:
        try:
            df2[c] = clean_numeric_series(df2[c]).fillna(0)
        except Exception:
            pass

    # populate standard existence columns if possible (best-effort)
    if 'existencia_cdpa' not in df2.columns:
        # try use first existence-like column name
        if existencia_cols:
            df2['total_existencias'] = df2[existencia_cols].sum(axis=1)
        else:
            df2['total_existencias'] = 0

    # monto existencias
    monto_candidates = [c for c in df2.columns if 'MONTO' in str(c).upper() or 'VALOR' in str(c).upper()]
    if monto_candidates:
        df2['monto_existencias'] = clean_numeric_series(df2[monto_candidates[0]]).fillna(0)
    else:
        df2['monto_existencias'] = 0

    # cast snapshot_date to ISO if present
    if 'snapshot_date' in df2.columns:
        df2['snapshot_date'] = pd.to_datetime(df2['snapshot_date'], errors='coerce')

    return df2


def write_clean_and_report(df_clean: pd.DataFrame, input_path: Path, out_dir: Path = CLEANED_DIR, reports_dir: Path = REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    out_path = out_dir / f"cleaned_{stem}.csv"
    # choose canonical columns to save if present
    save_cols = [c for c in CANONICAL_COLS if c in df_clean.columns]
    # Filter out rows that are not informative to downstream ML tasks.
    # A row is considered informative if at least one of:
    # - `precio_unitario` is present (not NaN)
    # - `total_existencias` != 0
    # - `monto_existencias` != 0
    # - `DESCRIPCION_LIMPIA` is non-empty and not purely numeric
    pre_count = int(len(df_clean))
    df_tmp = df_clean.copy()
    import pandas as _pd
    desc = _pd.Series([''] * len(df_tmp))
    if 'DESCRIPCION_LIMPIA' in df_tmp.columns:
        desc = df_tmp['DESCRIPCION_LIMPIA'].astype(str).fillna('').str.strip()
    desc_ok = desc.ne('') & ~desc.str.isnumeric()
    price_ok = False
    if 'precio_unitario' in df_tmp.columns:
        price_ok = df_tmp['precio_unitario'].notna()
    else:
        price_ok = _pd.Series([False] * len(df_tmp))
    total_exist = _pd.Series([0] * len(df_tmp))
    if 'total_existencias' in df_tmp.columns:
        total_exist = df_tmp['total_existencias'].fillna(0)
    monto_exist = _pd.Series([0] * len(df_tmp))
    if 'monto_existencias' in df_tmp.columns:
        monto_exist = df_tmp['monto_existencias'].fillna(0)

    informative_mask = (price_ok) | (total_exist != 0) | (monto_exist != 0) | (desc_ok)
    df_saved = df_tmp.loc[informative_mask].copy()
    dropped_count = int((~informative_mask).sum())

    # Save only canonical columns to keep outputs consistent
    df_saved.to_csv(out_path, index=False, encoding='utf-8', columns=save_cols)

    # write JSON report
    # compute simple checksum of written file for traceability
    def _file_md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    checksum = None
    try:
        checksum = _file_md5(out_path)
    except Exception:
        checksum = None

    report = {
        'input_file': str(input_path),
        'rows_input': pre_count,
        'rows_output': int(len(df_saved)),
        'rows_dropped_non_informative': dropped_count,
        'unique_presentations': int(df_saved['DESCRIPCION_LIMPIA'].nunique()) if 'DESCRIPCION_LIMPIA' in df_saved.columns else None,
        'columns_saved': save_cols,
        'output_checksum_md5': checksum
    }
    report_path = reports_dir / f"report_{stem}.json"
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return out_path


def run_single(input_file: Path) -> Optional[Path]:
    # check overrides
    stem = input_file.stem
    opt = None
    for k, v in OVERRIDES.items():
        if k in stem:
            opt = v
            break

    enc = None
    delim = None
    df = None
    # if override specifies encodings and sep, try them
    if opt is not None:
        # prefer explicit encoding list or single encoding
        enc_list = opt.get('encodings') or ([opt.get('encoding')] if opt.get('encoding') else None)
        sep = opt.get('sep') if 'sep' in opt else opt.get('delimiter')
        header = opt.get('header', 'infer')
        if enc_list:
            for e in enc_list:
                try:
                    df = pd.read_csv(input_file, header=header, encoding=e, sep=sep or ',', low_memory=False, quotechar='"', engine='python')
                    enc = e
                    delim = sep or ','
                    break
                except Exception:
                    df = None
            if df is None:
                # fallback detect
                enc, delim = detect_encoding_and_delim(input_file)
                try:
                    df = pd.read_csv(input_file, header=header, encoding=enc, sep=delim, low_memory=False)
                except Exception:
                    df = pd.read_csv(input_file, encoding='latin1', sep=',', low_memory=False)
        else:
            enc = opt.get('encoding') or detect_encoding_and_delim(input_file)[0]
            delim = opt.get('sep') or detect_encoding_and_delim(input_file)[1]
            try:
                df = pd.read_csv(input_file, header=opt.get('header', 'infer'), encoding=enc, sep=delim, low_memory=False)
            except Exception:
                df = pd.read_csv(input_file, encoding='latin1', sep=',', low_memory=False)
        # if expected columns specified and header was None, assign/truncate/pad
        if opt.get('expected_cols') and (opt.get('header') is None):
            cols = opt.get('expected_cols')
            if df.shape[1] >= len(cols):
                df = df.iloc[:, :len(cols)]
            else:
                for i in range(len(cols) - df.shape[1]):
                    df[df.shape[1] + i] = ''
            df.columns = cols
    else:
        # no specific overrides — detect encoding/delimiter
        enc, delim = detect_encoding_and_delim(input_file)
        try:
            df = pd.read_csv(input_file, encoding=enc, sep=delim, low_memory=False)
        except Exception:
            # fallback read with latin1 and comma
            df = pd.read_csv(input_file, encoding='latin1', sep=',', low_memory=False)
    df2 = normalize_dataframe(df, source_name=input_file.stem)

    # malformed check if requested by override
    malformed_info = None
    if opt and opt.get('malformed_check'):
        # perform a line-level field counting similar to legacy report
        txt = None
        try:
            # try to read raw bytes with encoding if known
            enc_try = enc or 'latin-1'
            txt = input_file.read_text(encoding=enc_try)
        except Exception:
            try:
                txt = input_file.read_text(encoding='latin-1')
            except Exception:
                txt = None
        if txt is not None:
            lines = txt.splitlines()
            def count_fields(line):
                in_q = False
                cnt = 0
                i = 0
                while i < len(line):
                    ch = line[i]
                    if ch == '"':
                        if i+1 < len(line) and line[i+1] == '"':
                            i += 2
                            continue
                        in_q = not in_q
                    elif ch == (delim or ',') and not in_q:
                        cnt += 1
                    i += 1
                return cnt + 1
            counts = [count_fields(l) for l in lines]
            c = Counter(counts)
            expected = c.most_common(1)[0][0] if c else None
            malformed = [(i+1, counts[i], lines[i]) for i in range(len(lines)) if counts[i] != expected]
            malformed_info = {
                'expected_fields': expected,
                'total_lines': len(lines),
                'malformed_count': len(malformed),
                'malformed_sample': [ {'line_no': idx, 'fields': fcnt, 'content': line[:400]} for idx, fcnt, line in malformed[:50] ]
            }
            # write text report
            try:
                out_txt = input_file.parent.parent / 'data' / 'cleaned' / f"{input_file.stem}_malformed_report.txt"
                out_txt.parent.mkdir(parents=True, exist_ok=True)
                with open(out_txt, 'w', encoding='utf-8') as fh:
                    fh.write("Expected fields per row: %s\n" % expected)
                    fh.write("Total lines: %d\n" % len(lines))
                    fh.write("Malformed lines: %d\n\n" % len(malformed))
                    fh.write('First malformed samples (line_no | fields | content):\n')
                    for idx, fcnt, line in malformed[:200]:
                        line_trunc = line[:400].replace('\n', ' ')
                        fh.write("%d | %d | %s\n" % (idx, fcnt, line_trunc))
            except Exception:
                pass

    out = write_clean_and_report(df2, input_file)

    # compare with existing legacy cleaned file if present
    legacy_path = input_file.parent / 'cleaned' / f"cleaned_{input_file.stem}.csv"
    compare_info = None
    try:
        if legacy_path.exists():
            try:
                legacy_df = pd.read_csv(legacy_path, encoding='utf-8', low_memory=False)
                # basic comparisons
                compare_info = {
                    'legacy_rows': int(len(legacy_df)),
                    'new_rows': int(len(df2)),
                    'legacy_unique_presentations': int(legacy_df['DESCRIPCION_LIMPIA'].nunique()) if 'DESCRIPCION_LIMPIA' in legacy_df.columns else None,
                    'new_unique_presentations': int(df2['DESCRIPCION_LIMPIA'].nunique()) if 'DESCRIPCION_LIMPIA' in df2.columns else None
                }
                # compute md5s for legacy file
                def _md5_path(p: Path):
                    h = hashlib.md5()
                    with open(p, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            h.update(chunk)
                    return h.hexdigest()
                compare_info['legacy_md5'] = _md5_path(legacy_path)
                compare_info['new_md5'] = out and (hashlib.md5(open(out,'rb').read()).hexdigest() if out else None)
            except Exception:
                compare_info = {'error': 'failed to read legacy file for comparison'}
    except Exception:
        compare_info = None

    # enrich report if malformed or compare info exist
    if malformed_info or compare_info:
        report_path = REPORTS_DIR / f"report_{input_file.stem}.json"
        try:
            with open(report_path, 'r', encoding='utf-8') as fh:
                rep = json.load(fh)
        except Exception:
            rep = {}
        if malformed_info:
            rep['malformed'] = malformed_info
        if compare_info:
            rep['legacy_comparison'] = compare_info
        with open(report_path, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"Wrote: {out}")
    return out


def run_all(input_dir: Path, pattern: str = "*.csv") -> None:
    input_dir = Path(input_dir)
    files = sorted(list(input_dir.glob(pattern)))
    for f in files:
        print("Processing:", f)
        try:
            run_single(f)
        except Exception as e:
            print(f"Failed {f}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', help='Single input CSV path')
    p.add_argument('--run-all', action='store_true', help='Process all CSVs in an input directory')
    p.add_argument('--input-dir', default=str(Path(__file__).resolve().parent.parent / 'data'), help='Directory to scan for CSVs')
    args = p.parse_args()

    if args.input:
        run_single(Path(args.input))
    elif args.run_all:
        run_all(Path(args.input_dir))
    else:
        print('Use --input <file> or --run-all')

if __name__ == '__main__':
    main()
