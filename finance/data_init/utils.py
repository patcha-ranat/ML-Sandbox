import json
import polars as pl


def convert_csv_to_json_lines(
    data_csv: list[str], relative_path: str = "finance/data"
) -> None:
    """
    Use polars to lazily read csv and sink to json lines format
    """
    for file in data_csv:
        filename_wo_extension = file.split(".")[0]
        lf = pl.scan_csv(f"{relative_path}/{file}")
        lf.sink_ndjson(f"{relative_path}/json/{filename_wo_extension}.json")


def convert_json_to_json_lines(
    data_json: list[str], relative_path: str = "finance/data"
) -> None:
    """
    Use polars to read json with DataFrame and write to json line format with specified schemas
    """
    for file in data_json:
        # read json file
        filename_wo_extension = file.split(".")[0]
        with open(f"{relative_path}/{file}", "r") as f:
            content: dict = json.load(f)
            f.close()

        # specify schema
        if filename_wo_extension == "mcc_codes":
            schema = {"mcc": content.keys(), "desc": content.values()}
        elif filename_wo_extension == "train_fraud_labels":
            content = content.get("target")
            schema = {"id": content.keys(), "is_fraud": content.values()}

        # export json according to schema
        df = pl.DataFrame(schema)
        df.write_ndjson(f"{relative_path}/json/{filename_wo_extension}.json")
