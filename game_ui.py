import base64
import signal
import time
import cv2
import httpx
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui
from game_css import css
from game_service import game, GameCounter

black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')
video_capture = cv2.VideoCapture(0)
game_counter = GameCounter()
ui.add_css(css)


def setup() -> None:
    video_image = ui.interactive_image().classes("player-video")
    ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))
    result_container = ui.card()
    counter_element = ui.card()
    ui.button('Play!', on_click=lambda: play_game(result_container, counter_element))

    app.on_shutdown(cleanup(video_capture))
    signal.signal(signal.SIGINT, handle_sigint)


@app.get('/video/frame')
async def grab_video_frame() -> Response:
    if not video_capture.isOpened():
        return placeholder

    _, frame = await run.io_bound(video_capture.read)

    if frame is None:
        return placeholder

    jpeg = await run.cpu_bound(convert, frame)
    return Response(content=jpeg, media_type='image/jpeg')


async def play_game(result_container, counter_element) -> None:
    await make_snapshot()
    results = game()
    game_counter.count(results[2])

    empty_container(counter_element)
    empty_container(result_container)
    populate_counter(counter_element)
    populate_result(result_container, results)


def populate_result(result_container, results):
    with result_container:
        ui.label(results[0])
        ui.label(results[1])
        ui.label(results[2])


def populate_counter(counter_element):
    with counter_element:
        ui.label(f"Player: {game_counter.player}")
        ui.label(f"Computer: {game_counter.computer}")


def empty_container(container):
    number_of_children = len(container.default_slot.children)
    for i in range(number_of_children):
        container.remove(0)  # List of children is update after every remove


async def make_snapshot() -> None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f'http://127.0.0.1:8080/video/frame')  # Adjust the URL as needed
            if response.status_code == 200:
                with open('player_sign.jpg', 'wb') as f:
                    f.write(response.content)
                print("Snapshot saved as 'snapshot.jpg'")
            else:
                print("Failed to capture snapshot:", response.status_code)
        except Exception as e:
            print("Error capturing snapshot:", e)


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


async def disconnect() -> None:
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


def handle_sigint(signum, frame) -> None:
    ui.timer(0.1, disconnect, once=True)
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup(video_capture) -> None:
    await disconnect()
    video_capture.release()


app.on_startup(setup)
ui.run()
