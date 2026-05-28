from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
from tqdm import tqdm


def glob_files(data_root: Path, mode: str, ending="parquet"):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.{}".format(ending)))
    return scenario_files


def preprocess(args):
    av1 = False
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        if av1:
            save_dir = data_root / "sharp_av1_processed" / mode
            #save_dir = Path("/home/ap1997/local/av1.1/DeMo_av1_processed") / mode
            from src.datamodules.av1_extractor import Av1Extractor
            extractor = Av1Extractor(save_path=save_dir, mode=mode)
        else:
            save_dir = data_root / "sharp_processed" / mode
            from src.datamodules.av2_extractor import Av2Extractor
            extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode, ending="csv" if av1 else "parquet")

        if args.parallel:
            with multiprocessing.Pool(16) as p:
                all_name = list(tqdm(p.imap(extractor.save, scenario_files), total=len(scenario_files)))
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--av1", action="store_true", help="Set this flag if preprocessing Argoverse 1.1 data instead of Argoverse 2.0 data")

    args = parser.parse_args()
    preprocess(args)
