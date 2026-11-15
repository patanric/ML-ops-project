"""Offline regression checks: python -m unittest test_inference_pipeline.py."""
import json
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

import inference_pipeline as pipeline
import serve
from pipeline_config import FG_NAME, FG_VERSION, LABEL_FG_NAME, LABEL_FG_VERSION

# get_batch_data(event_time=True) returns one prefixed event time per joined feature group
WEATHER_TIME = f"project_{FG_NAME}_{FG_VERSION}_time"
LABEL_TIME = f"project_{LABEL_FG_NAME}_{LABEL_FG_VERSION}_time"


class ForecastFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-11-15T12:00:30Z")
        clock = patch.object(pipeline.pd, "Timestamp", wraps=pd.Timestamp)
        clock.start().now.return_value = self.now
        self.addCleanup(clock.stop)
        request = patch.object(pipeline.requests, "get")
        request.start().return_value.json.return_value = {"current": {"cloud_cover": 99}}
        self.addCleanup(request.stop)
        self.fv, self.fg, self.clf = Mock(), MagicMock(), Mock()
        self.fg.time.__ge__.return_value = True
        self.model = Mock(version=1)

    def test_rejects_stale_future_and_empty_features_before_prediction(self):
        hour = self.now.floor("h")
        for times in ([hour - pd.Timedelta(hours=1)], [hour - pd.Timedelta(days=2)],
                      [hour + pd.Timedelta(hours=1)], []):
            with self.subTest(times=times):
                self.fv.get_batch_data.return_value = pd.DataFrame({
                    WEATHER_TIME: times, "cloud_cover": [20.] * len(times),
                })
                with self.assertRaisesRegex(pipeline.ForecastUnavailableError, "current hour"):
                    pipeline.predict(self.fv, self.fg, self.model, self.clf)
                self.clf.predict.assert_not_called()
                self.clf.predict_proba.assert_not_called()
                self.fg.select.assert_not_called()

    def test_current_hour_uses_live_cloud_cover_only_for_latest_row(self):
        hour = self.now.floor("h")
        times = [hour - pd.Timedelta(hours=1), hour]
        self.fv.get_batch_data.return_value = pd.DataFrame({
            WEATHER_TIME: [t.tz_localize(None) for t in reversed(times)],
            LABEL_TIME: [None, times[0].tz_localize(None)],  # newest hour has no label row yet
            "cloud_cover": [30., 20.],
        })
        self.fg.select.return_value.filter.return_value.read.return_value = pd.DataFrame({
            "time": times, "rain_next_2h": [0, None],
        })
        self.clf.predict.return_value = [0, 1]
        self.clf.predict_proba.return_value = pd.DataFrame([[.8, .2], [.2, .8]]).to_numpy()

        html = pipeline.predict(self.fv, self.fg, self.model, self.clf)

        features = self.clf.predict.call_args.args[0]
        self.assertEqual(features["cloud_cover"].tolist(), [20., 99.])
        self.assertEqual(features.columns.tolist(), ["cloud_cover"])  # no event time of either group
        records = json.loads(html.split("const DATA = ")[1].split(";")[0])
        self.assertEqual(records[-1]["time"], "2026-11-15T12:00:00Z")
        self.assertEqual(records[-1]["pred"], 1)
        self.assertIsNone(records[-1]["actual"])

    def test_web_returns_503_when_forecast_is_unavailable(self):
        handler = Mock(path="/")
        error = pipeline.ForecastUnavailableError("Current forecast unavailable")
        with patch.object(serve, "CONTEXT", (), create=True), \
                patch.object(serve, "predict", side_effect=error):
            serve.Handler.do_GET(handler)
        handler.send_error.assert_called_once_with(503, explain=str(error))
        handler.send_response.assert_not_called()
        handler.wfile.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
