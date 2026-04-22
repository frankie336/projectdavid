import os
from typing import Optional

from projectdavid_common import UtilsInterface, ValidationInterface

from projectdavid.clients.base_client import BaseAPIClient

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()


class TrainingClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        training_url: Optional[str] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key)

        # Training routes are behind the same nginx proxy as the core API.
        # Use base_url as the default — no separate training_url needed
        # unless explicitly overridden via TRAINING_BASE_URL.
        resolved_url = (
            training_url
            or os.getenv("TRAINING_BASE_URL")
            or base_url
            or "http://localhost:80"
        )
        self.training_url = resolved_url.rstrip("/")

    def create(
        self,
        dataset_id: str,
        base_model: str,
        framework: str = "unsloth",
        config: validator.TrainingConfig | dict | None = None,
    ) -> validator.TrainingJobRead:
        """
        Create and queue a training job.

        `config` accepts either a dict of raw parameters or a TrainingConfig
        instance (recommended — gets IDE autocomplete + local Pydantic
        validation before the HTTP round-trip).
        """
        if isinstance(config, validator.TrainingConfig):
            config_payload = config.model_dump(exclude_none=True)
        else:
            config_payload = config or {}

        payload = {
            "dataset_id": dataset_id,
            "base_model": base_model,
            "framework": framework,
            "config": config_payload,
        }
        response = self.client.post(
            f"{self.training_url}/v1/training-jobs/", json=payload
        )
        response.raise_for_status()
        return validator.TrainingJobRead.model_validate(response.json())

    def retrieve(self, job_id: str) -> validator.TrainingJobRead:
        response = self.client.get(f"{self.training_url}/v1/training-jobs/{job_id}")
        response.raise_for_status()
        return validator.TrainingJobRead.model_validate(response.json())

    def cancel(self, job_id: str) -> validator.TrainingJobCancelResponse:
        """
        Cancel a training job.

        Idempotent — calling cancel on a job in a terminal state (completed,
        failed, cancelled) returns the current status without error.

        For in-progress jobs, cancellation is asynchronous. The response status
        will be 'cancelling'; poll retrieve(job_id) to observe the transition
        to 'cancelled' after the worker unwinds the subprocess (typically
        within 30 seconds).

        Partial training artifacts are discarded on cancellation.
        """
        response = self.client.post(
            f"{self.training_url}/v1/training-jobs/{job_id}/cancel"
        )
        response.raise_for_status()
        return validator.TrainingJobCancelResponse.model_validate(response.json())

    # ------------------------------------------------------------------
    # WAIT FOR COMPLETION
    # ------------------------------------------------------------------
    def wait_for_completion(
        self,
        job_id: str,
        *,
        on_progress=None,
        poll_interval: float = 10.0,
        timeout: float = 7200.0,
    ) -> validator.TrainingJobRead:
        """
        Block until the training job reaches a terminal state.

        Terminal states: 'completed', 'failed', 'cancelled'.

        An optional `on_progress` callback receives every distinct metrics
        snapshot as it arrives (keyed by the 'step' field). Use it to drive
        live progress display without coupling the SDK to any particular
        output format.

        Example:
            def show(m):
                print(f"step={m.get('step')} loss={m.get('loss')}")

            job = client.training.wait_for_completion(job.id, on_progress=show)

        Raises TimeoutError if the job has not reached a terminal state
        within `timeout` seconds. Does NOT raise on a 'failed' or 'cancelled'
        job — callers inspect the returned job.status themselves, since a
        failed job is often still interesting (last_error, partial metrics).
        """
        import time  # local import keeps top-of-file imports minimal

        TERMINAL_STATES = {"completed", "failed", "cancelled"}
        deadline = time.monotonic() + timeout
        last_step = -1

        while time.monotonic() < deadline:
            job = self.retrieve(job_id)

            metrics = getattr(job, "metrics", None) or {}
            step = metrics.get("step")
            if on_progress is not None and step is not None and step != last_step:
                last_step = step
                try:
                    on_progress(metrics)
                except Exception as cb_err:
                    # Callback failures must not break polling.
                    logging_utility.warning(
                        "on_progress callback raised %s: %s",
                        type(cb_err).__name__,
                        cb_err,
                    )

            if job.status in TERMINAL_STATES:
                return job

            time.sleep(poll_interval)

        raise TimeoutError(f"Training job {job_id} did not complete within {timeout}s")

    # ------------------------------------------------------------------
    # DIAGNOSTIC PEEK (Secure Multi-tenant Gateway)
    # ------------------------------------------------------------------
    def peek_queue(self) -> validator.TrainingQueueList:
        """
        Securely check the remote Redis queue via the Training API to see
        pending jobs belonging only to the current user.
        """
        response = self.client.get(f"{self.training_url}/v1/training-jobs/queue/peek")
        response.raise_for_status()
        return validator.TrainingQueueList.model_validate(response.json())
