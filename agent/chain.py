class _Stop:
    def __init__(self, value):
        self.value = value


class Chain:
    def __init__(self, steps):
        self.steps = list(steps)

    def __or__(self, other):
        extra = other.steps if isinstance(other, Chain) else [other]
        return Chain(self.steps + extra)

    def invoke(self, data):
        for step in self.steps:
            data = step(data)
            if isinstance(data, _Stop):
                return data.value
        return data


def step(fn):
    return Chain([fn])


def stop(value):
    return _Stop(value)
