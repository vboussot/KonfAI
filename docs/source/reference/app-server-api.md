# App server HTTP API

This page summarizes the FastAPI endpoints implemented in `konfai_apps.app_server`.

Authentication is enforced through bearer tokens when the server runs with the
default `--auth bearer` mode.

## Health and system endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check. |
| `GET` | `/available_devices` | Visible GPU ids and names. |
| `GET` | `/ram` | Server RAM usage. |
| `GET` | `/vram` | VRAM usage for requested devices. |

## App repository endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/repo_apps_list` | List configured apps. |
| `GET` | `/repo_apps/{app_id}` | Fetch app metadata and capabilities. |
| `GET` | `/repo_apps_config/{app_id}` | Download app config files as a zip archive. |

## Job submission endpoints

| Method | Path |
| --- | --- |
| `POST` | `/apps/{app_name}/infer` |
| `POST` | `/apps/{app_name}/evaluate` |
| `POST` | `/apps/{app_name}/uncertainty` |
| `POST` | `/apps/{app_name}/pipeline` |
| `POST` | `/apps/{app_name}/fine_tune` |

All job-submission endpoints return:

```json
{
  "job_id": "abc123",
  "status_url": "/jobs/abc123",
  "logs_url": "/jobs/abc123/logs",
  "result_url": "/jobs/abc123/result"
}
```

## Job control endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/jobs/{job_id}` | Job metadata and status. |
| `GET` | `/jobs/{job_id}/logs` | Server-sent event log stream. |
| `GET` | `/jobs/{job_id}/result` | Download the zipped result bundle. |
| `POST` | `/jobs/{job_id}/kill` | Terminate a running job. |

## Notes on behavior

The following behavior is visible directly in the server code:

- uploads are written into isolated temporary workspaces
- jobs are scheduled with optional GPU allocation
- logs are streamed over SSE
- outputs are zipped before download
- workspaces are deleted after a grace period

## See also

- {doc}`../usage/remote-server`
- {doc}`../concepts/apps`
