"""youtube.py second pass (iteration 60, DeepSeek riders). Red first."""

from __future__ import annotations

from searchio.providers.youtube import _videos_from_data


def _card(**over):
    r = {
        "videoId": "abc123def45",
        "title": {"runs": [{"text": "A video"}]},
        "ownerText": {"runs": [{"text": "Chan", "navigationEndpoint": {"browseEndpoint": {"browseId": "UC1"}}}]},
        "publishedTimeText": {"simpleText": "3 days ago"},
        "viewCountText": {"simpleText": "1,234 views"},
        "lengthText": {"simpleText": "3:45"},
    }
    r.update(over)
    return {"contents": [{"videoRenderer": r}]}


class TestOddRenderersDoNotKillThePage:
    def test_empty_owner_runs(self):
        # DeepSeek youtube #2: ownerText.runs == [] indexed [0] -> IndexError
        # and the WHOLE results page parsed to nothing.
        docs = _videos_from_data(_card(ownerText={"runs": []}))
        assert len(docs) == 1 and docs[0].meta.get("channel_id", "") == ""

    def test_owner_runs_not_a_list(self):
        docs = _videos_from_data(_card(ownerText={"runs": "garbage"}))
        assert len(docs) == 1


class TestSnippetCarriesTheDescription:
    def test_detailed_metadata_snippet_is_included(self):
        # DeepSeek youtube #5: the snippet was channel/views/age chrome only;
        # the card's description snippet (what the agent needs to judge the
        # video) was dropped.
        docs = _videos_from_data(_card(detailedMetadataSnippets=[
            {"snippetText": {"runs": [{"text": "How to "}, {"text": "replace a bike chain", "bold": True}, {"text": " in 5 minutes"}]}}]))
        assert "replace a bike chain" in docs[0].snippet
        assert docs[0].meta.get("description", "").startswith("How to replace")
