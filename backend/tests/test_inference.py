"""Tests for inference service logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSoftmax:
    def test_softmax_output_sums_to_one(self):
        from backend.app.services.inference import _softmax

        logits = np.array([2.0, 1.0, 0.1])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_all_positive(self):
        from backend.app.services.inference import _softmax

        logits = np.array([-1.0, 0.0, 1.0])
        probs = _softmax(logits)
        assert (probs > 0).all()


class TestGetLabelName:
    def test_label_name_found(self):
        from backend.app.services.inference import _get_label_name

        label_map = {"0": "activate_my_card", "5": "check_balance"}
        assert _get_label_name(0, label_map) == "activate_my_card"

    def test_label_name_fallback(self):
        from backend.app.services.inference import _get_label_name

        assert _get_label_name(99, {}) == "class_99"


class TestPredictSingle:
    @pytest.mark.asyncio
    async def test_predict_single_returns_result(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        # Return logits for 3 classes
        mock_session.run.return_value = [np.array([[2.0, 0.5, -1.0]])]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_model = {
            "session": mock_session,
            "tokenizer": mock_tokenizer,
            "backend": "onnx",
            "label_map": {"0": "class_a", "1": "class_b", "2": "class_c"},
        }

        with patch("backend.app.services.inference.get_model", return_value=mock_model):
            from backend.app.services.inference import predict_single

            result = await predict_single("test text")
            assert result.label == 0
            assert result.confidence > 0
            assert result.label_name == "class_a"
            assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_confidence_sum_to_one(self):
        """Softmax probabilities should sum to ~1."""
        from backend.app.services.inference import _softmax

        logits = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_label_in_valid_range(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        n_classes = 77
        logits = np.random.randn(1, n_classes)
        mock_session.run.return_value = [logits]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_model = {
            "session": mock_session,
            "tokenizer": mock_tokenizer,
            "backend": "onnx",
            "label_map": {},
        }

        with patch("backend.app.services.inference.get_model", return_value=mock_model):
            from backend.app.services.inference import predict_single

            result = await predict_single("test")
            assert 0 <= result.label < n_classes


class TestPredictBatch:
    @pytest.mark.asyncio
    async def test_predict_batch_returns_list(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_session.run.return_value = [np.array([[2.0, 0.5, -1.0]])]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_model = {
            "session": mock_session,
            "tokenizer": mock_tokenizer,
            "backend": "onnx",
            "label_map": {},
        }

        with patch("backend.app.services.inference.get_model", return_value=mock_model):
            from backend.app.services.inference import predict_batch

            results = await predict_batch(["text1", "text2"])
            assert isinstance(results, list)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_prediction_result_fields(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
        mock_session.run.return_value = [np.array([[1.0, 0.0]])]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])}

        mock_model = {"session": mock_session, "tokenizer": mock_tokenizer, "backend": "onnx", "label_map": {}}

        with patch("backend.app.services.inference.get_model", return_value=mock_model):
            from backend.app.services.inference import predict_single

            result = await predict_single("test")
            assert hasattr(result, "label")
            assert hasattr(result, "confidence")
            assert hasattr(result, "label_name")
            assert hasattr(result, "latency_ms")


class TestInferenceDeterministic:
    @pytest.mark.asyncio
    async def test_inference_deterministic(self):
        """Same input with same model should produce same output."""
        fixed_logits = np.array([[2.0, 0.5, -1.0]])
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
        mock_session.run.return_value = [fixed_logits]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": np.array([[1, 2]]), "attention_mask": np.array([[1, 1]])}

        mock_model = {"session": mock_session, "tokenizer": mock_tokenizer, "backend": "onnx", "label_map": {}}

        with patch("backend.app.services.inference.get_model", return_value=mock_model):
            from backend.app.services.inference import predict_single

            r1 = await predict_single("test")
            r2 = await predict_single("test")
            assert r1.label == r2.label
            assert r1.confidence == r2.confidence
