#!/usr/bin/env python3
"""Renew Drive OAuth credentials using a local browser callback; never print tokens."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-secret', type=Path, default=root / 'client_secret.json')
    parser.add_argument('--output', type=Path, default=root / 'drive_token.json')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secret), scopes=['https://www.googleapis.com/auth/drive'])
    credentials = flow.run_local_server(
        host='localhost', bind_addr='127.0.0.1', port=args.port,
        open_browser=False, timeout_seconds=900, access_type='offline', prompt='consent',
        authorization_prompt_message='Open this URL in your browser:\n{url}',
        success_message='Drive authorization received. You may close this window.')
    if not credentials.refresh_token:
        raise RuntimeError('No refresh token returned; existing token file was not changed')
    credentials.refresh(Request())
    drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    drive.files().list(pageSize=1, fields='files(id)').execute(num_retries=3)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.drive-token-', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(credentials.to_json())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Drive refresh and API access verified; credentials saved to {output}', flush=True)


if __name__ == '__main__':
    main()
