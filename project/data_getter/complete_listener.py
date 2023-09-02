from typing import Callable

events : dict[str, list[Callable]] = {}

def subscribe(event_code:str, fn:Callable):
    if event_code not in events:
        events[event_code] = []
    events[event_code].append(fn)


def announce(event_code:str, msg:str):
    functions_listening = events.get(event_code, [])

    for fn in functions_listening:
        fn(msg)
