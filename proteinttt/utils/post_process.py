import ast


def str_to_dict(s):
    d = {}
    pairs = s.split(',')
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':', 1)
            d[key.strip()] = value.strip()
    return d


def parse_log(file):
    id_seq = file.split("_")[0]
    output_path = base_path / 'esm_fold' / f'{id_seq}.pdb'
    if output_path.exists():
        return

    log = pd.read_csv(base_path / "logs" / file, sep="\t")
    s = log['ttt_step_data'].iloc[0]
    result = str_to_dict(s)
    txt = result['{0']

    data = ast.literal_eval(txt)
    pdb_str = data["eval_step_preds"]["pdb"]

    # write to PDB
    # with open(base_path / 'esm_fold' / f'{id_seq}.pdb', "w", encoding="utf-8") as f:
    #     f.write(pdb_str.rstrip("\n") + "\n")
    #     if not pdb_str.strip().endswith(("END", "ENDMDL")):
    #         f.write("END\n")

    output_path.write_text(pdb_str)

