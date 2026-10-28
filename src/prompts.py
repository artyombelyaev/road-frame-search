# Тексты запросов для CLIP. Шаблоны сравниваю на train_sub, на eval идёт один.
# Фразы для значений выбраны до первых метрик и не подбирались. После EDA избегаю "clear sky"
# (ночью clear значит скорее "без осадков") и "snowing" (snowy - чаще снег на обочине).
# tags - как пишет человек: сырые метки через запятую, время, погода, сцена.
# sentence - фраза из словаря ниже, dashcam - та же фраза с префиксом про видеорегистратор.

from . import config

TEMPLATES = ["tags", "sentence", "dashcam"]

TAG_ORDER = ["timeofday", "weather", "scene"]

SCENE = {
    "city street": "a city street",
    "highway": "a highway",
    "residential": "a residential street",
    "parking lot": "a parking lot",
    "tunnel": "a tunnel",
    "gas stations": "a gas station",
}
WEATHER = {
    "clear": "in clear weather",
    "overcast": "under an overcast sky",
    "partly cloudy": "under a partly cloudy sky",
    "rainy": "in the rain",
    "snowy": "with snow",
    "foggy": "in fog",
}
TIME = {
    "daytime": "during the day",
    "night": "at night",
    "dawn/dusk": "at dawn or dusk",
}


def query_text(q, template: str) -> str:
    values = {a: q[a] for a in config.ATTRS if isinstance(q.get(a), str)}
    if template == "tags":
        return ", ".join(values[a] for a in TAG_ORDER if a in values)
    scene = SCENE[values["scene"]] if "scene" in values else "a road"
    parts = [scene]
    if "weather" in values:
        parts.append(WEATHER[values["weather"]])
    if "timeofday" in values:
        parts.append(TIME[values["timeofday"]])
    prefix = "a photo of" if template == "sentence" else "a dashcam photo of"
    return f"{prefix} {' '.join(parts)}"
