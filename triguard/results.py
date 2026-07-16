import csv
import os


def _serialized_csv_row(row, header):
    return {
        field: "" if row.get(field) is None else str(row.get(field))
        for field in header
    }


def csv_contains_identity(path, identity):
    if not os.path.exists(path):
        return False
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(identity) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Existing CSV {path} lacks identity columns: {sorted(missing)}"
            )
        expected = {
            key: "" if value is None else str(value) for key, value in identity.items()
        }
        return any(
            all(row.get(key, "") == expected[key] for key in expected)
            for row in reader
        )


def append_csv(path, row, header, key_fields=None, duplicate_policy="error"):
    """Append one result while rejecting schema drift and duplicate identities."""
    if duplicate_policy not in {"error", "skip"}:
        raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")
    if len(set(header)) != len(header):
        raise ValueError("CSV header contains duplicate columns.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    key_fields = list(key_fields or [])
    missing_keys = [
        field for field in key_fields if field not in header or field not in row
    ]
    if missing_keys:
        raise ValueError(f"Missing CSV identity fields: {missing_keys}")
    if exists:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader) if key_fields else []
        if existing_header != list(header):
            raise ValueError(
                "CSV schema mismatch for "
                f"{path}. Existing columns do not match this run. "
                "Use a new --out directory or remove the old CSV file."
            )
        serialized = _serialized_csv_row(row, header)
        duplicate = next(
            (
                existing
                for existing in existing_rows
                if all(
                    existing.get(field, "") == serialized[field]
                    for field in key_fields
                )
            ),
            None,
        )
        if duplicate is not None:
            if duplicate_policy == "skip":
                return False
            identity = ", ".join(
                f"{field}={serialized[field]}" for field in key_fields
            )
            raise ValueError(
                f"Duplicate experiment row in {path}: {identity}. "
                "Use a new configuration or remove the existing row deliberately."
            )
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return True
