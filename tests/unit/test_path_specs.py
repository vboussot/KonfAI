from konfai.utils.utils import split_path_spec


def test_split_path_spec_supports_unix_style_dataset_specs() -> None:
    assert split_path_spec("./Dataset") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:mha") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:a:mha", allowed_flags={"a", "i"}) == ("./Dataset", "a", "mha")
    assert split_path_spec("./Predictions/TRAIN_01/Dataset:i:mha", allowed_flags={"a", "i"}) == (
        "./Predictions/TRAIN_01/Dataset",
        "i",
        "mha",
    )


def test_split_path_spec_supports_windows_paths_without_breaking_drive_letters() -> None:
    assert split_path_spec(r"C:\Dataset") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:mha") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:a:mha", allowed_flags={"a", "i"}) == (r"C:\Dataset", "a", "mha")
