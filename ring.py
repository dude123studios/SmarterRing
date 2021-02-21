import os

import json
from pathlib import Path
from pprint import pprint

from ring_doorbell import Ring, Auth
from oauthlib.oauth2 import MissingTokenError

from video_loader import getFirstFrame
from resources.face_recognition import who_is_here

cache_file = Path("test_token.cache")


def token_updated(token):
    cache_file.write_text(json.dumps(token))


def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code


def main():
    if cache_file.is_file():
        auth = Auth("MyProject/1.0", json.loads(cache_file.read_text()), token_updated)
    else:
        username = os.environ.get('USERNAME')
        password = os.environ.get('PASSWORD')
        auth = Auth("MyProject/1.0", None, token_updated)
        try:
            auth.fetch_token(username, password)
        except MissingTokenError:
            auth.fetch_token(username, password, otp_callback())

    ring = Ring(auth)
    ring.update_data()

    devices = ring.devices()
    pprint(devices)
    handle = handle_video(ring)
    if handle:
        print(handle)

def handle_video(ring):
    devices = ring.devices()
    doorbell = devices['authorized_doorbots'][0]
    doorbell.recording_download(
        doorbell.history(limit=50, kind='ding')[0]['id'],
        filename='last_ding.mp4',
        override=True)
    frame = getFirstFrame('last_ding.mp4')
    os.remove('last_ding.mp4')
    people = who_is_here(frame)
    if people == 'ERROR':
        print('[WARNING] No faces were detected, something might be blocking their faces,' +
              ' or the people could have turned around')
        return None
    return_string = ''
    for person in people:
        return_string = return_string + person + ', '
    return_string = return_string + 'Are/Is at the door!'
    return return_string
