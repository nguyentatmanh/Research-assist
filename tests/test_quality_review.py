import pytest
from pydantic import ValidationError

from research_assist.experiments.quality_review import Review, apply_reviews, review_template


def sample():
    return {
        "cases": [
            {"id": "a", "question": "Question A", "answer_sha256": "hash_a"},
            {"id": "b", "question": "Question B", "answer_sha256": "hash_b"},
        ],
        "summary": {"cases": 2},
    }


def review(**changes):
    return Review.model_validate(
        {
            "case_id": "a",
            "answer_sha256": "hash_a",
            "reviewer": "independent reviewer",
            "faithfulness": 0.8,
            "completeness": 0.6,
            **changes,
        }
    )


def test_scores_bound_to_exact_answer_and_unreviewed_cases_not_counted_as_zero():
    original = sample()
    result = apply_reviews(original, [review()])
    assert result["summary"]["human_faithfulness"] == 0.8
    assert result["summary"]["human_reviewed_cases"] == 1
    assert "human_review" not in original["cases"][0]
    assert review_template(original)[0]["faithfulness"] is None


@pytest.mark.parametrize("changes", [{"case_id": "foreign"}, {"answer_sha256": "old_hash"}])
def test_foreign_or_stale_reviews_rejected(changes):
    with pytest.raises(ValueError):
        apply_reviews(sample(), [review(**changes)])


def test_duplicate_scores_rejected():
    with pytest.raises(ValueError):
        apply_reviews(sample(), [review(), review()])


def test_empty_reviewer_and_out_of_range_scores_rejected():
    with pytest.raises(ValidationError):
        review(reviewer="")
    with pytest.raises(ValidationError):
        review(faithfulness=1.1)
