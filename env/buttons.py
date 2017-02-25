from env.action import Action, get_action_code

class Buttons:
    def __init__(self):
        self._action = Action.FLAT
        self._esc_hit = False
        self._long = False
        self._short = False

    def on_press_esc(self):
        self._esc_hit = True

    def on_press_up(self):
        self._long = True
        self._calc_action()

    def on_release_up(self):
        self._long = False
        self._calc_action()

    def on_press_down(self):
        self._short = True
        self._calc_action()

    def on_release_down(self):
        self._short = False
        self._calc_action()

    def _calc_action(self):
        if self._long and not self._short:
            self._action = Action.BUY
        elif self._short and not self._long:
            self._action = Action.SELL
        else:
            self._action = Action.FLAT

    @property
    def esc_hit(self):
        return self._esc_hit

    @property
    def action(self):
        return get_action_code(self._action)


_buttons = Buttons()

def get_buttons() ->Buttons:
    return _buttons