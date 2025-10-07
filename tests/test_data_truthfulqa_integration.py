import pytest
from halu.data.truthfulqa import TruthfulQADataset
from halu.data.base import LETTERS
from halu.data.truthfulqa import TruthfulQADatasetRow

pytestmark = pytest.mark.hf

def test_truthfulqa_hf_adapter_smoke_and_invariants():
    # Requires: pip install datasets
    ds = TruthfulQADataset()

    # Pull a few examples through the Dataset interface (like the pipeline does)
    # Using a small sample keeps the test fast yet realistic.
    examples = list(ds.iter_examples(sample_size=8, seed=1337))
    assert len(examples) > 0

    for ex in examples:
        # Basic shape checks
        assert ex.qid
        assert isinstance(ex.question, str) and len(ex.question) > 0
        assert len(ex.options) >= 2

        # Labels should be unique and in LETTERS
        labels = [o.label for o in ex.options]
        assert len(set(labels)) == len(labels)
        assert all(L in LETTERS for L in labels)

        # Gold must be one of the labels
        assert ex.gold_letter in labels

def _row(choices, labels):
    return {
        "question": "Q?",
        "mc1_targets": {"choices": choices, "labels": labels},
    }

def test_row_to_mcq_no_shuffle_maps_labels_stably():
    row = _row(["a","b","c"], [0,1,0])
    ex = TruthfulQADatasetRow(row).row_to_mcq("7")
    assert [o.label for o in ex.options] == LETTERS[:3]
    assert ex.options[1].text == "b"
    assert ex.gold_letter == "B"

def test_row_to_mcq_shuffle_is_deterministic_per_qid_and_seed():
    row = _row(["a","b","c","d"], [0,0,1,0])
    r1 = TruthfulQADatasetRow(row, shuffle_options=True, shuffle_seed=42).row_to_mcq("10")
    r2 = TruthfulQADatasetRow(row, shuffle_options=True, shuffle_seed=42).row_to_mcq("10")
    assert [o.text for o in r1.options] == [o.text for o in r2.options]
    assert r1.gold_letter == r2.gold_letter

def test_row_to_mcq_shuffle_gold_tracks_correct_text():
    row = _row(["a","b","c","d"], [0,0,1,0])  # "c" correct
    ex = TruthfulQADatasetRow(row, shuffle_options=True, shuffle_seed=0).row_to_mcq("1")
    # Find the option whose text is "c"; its label must equal gold_letter
    label_of_c = next(L for L,t in [(o.label,o.text) for o in ex.options] if t=="c")
    assert ex.gold_letter == label_of_c