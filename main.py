import argparse

from gensim.models import Word2Vec

from src.bilstm_pipeline import run_bilstm_pipeline
from src.classical_pipeline import run_classical_pipeline
from src.config import ProjectConfig
from src.data_utils import describe_datasets, load_competition_data
from src.feature_engineering import train_word2vec
from src.preprocess import preprocess_dataframe
from src.utils import ensure_directories, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Word2Vec NLP Tutorial project runner")
    parser.add_argument(
        "--include-bilstm",
        action="store_true",
        help="Also train the optional Word2Vec + BiLSTM pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    ensure_directories(
        [
            config.artifacts_dir,
            config.reports_dir,
            config.plots_dir,
            config.submissions_dir,
        ]
    )
    set_seed(config.seed)

    labeled_df, unlabeled_df, test_df = load_competition_data(config.data_dir)
    summary = describe_datasets(labeled_df, unlabeled_df, test_df)
    save_json(config.reports_dir / "dataset_summary.json", summary)

    print("Dataset summary:", summary)
    print("Preprocessing labeled, unlabeled, and test reviews...")
    labeled_df = preprocess_dataframe(labeled_df)
    unlabeled_df = preprocess_dataframe(unlabeled_df)
    test_df = preprocess_dataframe(test_df)

    word2vec_path = config.artifacts_dir / "word2vec.model"
    if word2vec_path.exists():
        print(f"Loading cached Word2Vec model from {word2vec_path}")
        word2vec_model = Word2Vec.load(str(word2vec_path))
    else:
        print("Training Word2Vec on labeled + unlabeled reviews...")
        unsupervised_corpus = labeled_df["tokens"].tolist() + unlabeled_df["tokens"].tolist()
        word2vec_model = train_word2vec(unsupervised_corpus, config)
        word2vec_model.save(str(word2vec_path))
        print(f"Word2Vec model saved to {word2vec_path}")

    classical_result = run_classical_pipeline(labeled_df, test_df, word2vec_model, config)
    print("Classical pipeline finished.")
    print("Best model:", classical_result["best_model"])
    print("Validation metrics:", classical_result["validation_metrics"])
    print("Submission:", classical_result["submission_path"])

    if args.include_bilstm:
        print("Training optional BiLSTM pipeline...")
        bilstm_result = run_bilstm_pipeline(labeled_df, test_df, word2vec_model, config)
        print("BiLSTM validation metrics:", bilstm_result["validation_metrics"])
        print("BiLSTM submission:", bilstm_result["submission_path"])


if __name__ == "__main__":
    main()
