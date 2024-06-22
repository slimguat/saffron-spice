from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable
import numpy as np
import datetime
import sys
from pathlib import Path
from ..fit_models import flat_inArg_multiGauss
from .utils import find_nth_occurrence


class LockProtocols:
    def __init__(self):
        self.lock = {}
        # self.block = []
        # self.line_nmaes = None
        self.fit_func = None
        # self.unlocked_init_params = []
        # self.unlocked_quentities = []
        # self.locked_init_params = []
        # self.locked_quentities = []
        self.import_function_list = []
        self._dir_tmp_functions = None

    def add_lock(self, target_line_ind, lock_line_ind, wvl):
        if not isinstance(lock_line_ind, Iterable):
            lock_line_ind = [lock_line_ind]
        if not isinstance(wvl, Iterable):
            wvl = [wvl]

        if self.lock:
            if any(tgt in lock_line_ind for tgt in list(self.lock.keys())):
                raise Exception(f"A target line cannot be used as lock line")
            for target, lock_option in self.lock.items():
                if target_line_ind in lock_option:
                    raise Exception(
                        f"{target_line_ind} is already a lock option groupe them under same target please"
                    )
                for existing_lock_line in [tab[0] for tab in lock_option]:
                    if existing_lock_line in lock_line_ind:
                        raise Exception(
                            f"The value  Does exist already delete it first in one lock options"
                        )

        if not target_line_ind in self.lock.keys():
            self.lock[target_line_ind] = []

        for ind, new_lock_line in enumerate(lock_line_ind):
            self.lock[target_line_ind].append([new_lock_line, wvl[ind]])

    def add_block(self, target_line_ind, wvl):
        if not isinstance(target_line_ind, Iterable):
            target_line_ind = [target_line_ind]
        if not isinstance(wvl, Iterable):
            wvl = [wvl]

        if any(item in target_line_ind for item in [tab[0] for tab in self.block]):
            raise Exception("can't block a line in more than one line")
        for i in range(len(target_line_ind)):
            self.block.append([target_line_ind[i], wvl[i]])

    def __repr__(self):
        return (
            str(self.lock)
            + "\n"
            # str(self.block)+"\n"
            # str(self.unlocked_init_params)+"\n"+
            # str(self.unlocked_quentities)+"\n"+
            # str(self.locked_init_params)+"\n"+
            # str(self.locked_quentities)
        )


def gen_locked_params(
    unlocked_init_params, unlocked_quentities, lock_protocols, unlocked_cov=None
):
    assert not (
        len(unlocked_init_params) == 0
        or len(unlocked_quentities) == 0
        or len(unlocked_init_params) != len(unlocked_quentities) == 0
    )

    code_lock2unlock = "[" + "," * len(unlocked_init_params) + "]"
    code_unlock2lock = "[" + "," * len(unlocked_init_params) + "]"
    unlocked_init_params = list(unlocked_init_params)
    if not unlocked_cov is None:
        unlocked_cov = list(unlocked_cov)
    locked_init_params = unlocked_init_params.copy()
    if not unlocked_cov is None:
        locked_cov = unlocked_cov.copy()
    locked_quentities = unlocked_quentities.copy()
    lines_done = []
    for target, lines in lock_protocols.lock.items():
        for line in lines:
            ind_line = find_nth_occurrence(locked_quentities, "I", line[0] + 1) + 1
            locked_quentities.pop(ind_line)
            locked_init_params.pop(ind_line)
            if not unlocked_cov is None:
                locked_cov.pop(ind_line)
            # code initiation as string (this way the function will be executing faster)
            code_unlock2lock = code_unlock2lock[0:1] + code_unlock2lock[2:]
            ind_line4unlock_code = (
                find_nth_occurrence(unlocked_quentities, "I", line[0] + 1) + 1
            )
            ind_target = find_nth_occurrence(unlocked_quentities, "I", target + 1) + 1
            ind_comma = find_nth_occurrence(
                code_lock2unlock, ",", ind_line4unlock_code + 1
            )
            ind_target2 = (
                ind_target
                - len([lin for lin in lines_done if line > lin])
                + unlocked_quentities[:ind_target].count("B")
            )
            code_lock2unlock = (
                code_lock2unlock[:ind_comma]
                + "{0}"
                + "["
                + str(ind_target2)
                + "]"
                + "+"
                + str(line[1])
                + code_lock2unlock[ind_comma:]
            )
            lines_done.append(line)
    # filling the rest of the code with usual array values
    iter = 1
    ind_array = 0
    code_lock2unlock = "," + code_lock2unlock[1:]
    code_unlock2lock = "," + code_unlock2lock[1:]
    while iter < code_lock2unlock.count(","):
        ind_beg1 = find_nth_occurrence(code_lock2unlock, ",", iter)
        ind_end1 = find_nth_occurrence(code_lock2unlock, ",", iter + 1)
        ind_beg2 = find_nth_occurrence(code_unlock2lock, ",", ind_array + 1)
        ind_end2 = find_nth_occurrence(code_unlock2lock, ",", ind_array + 2)

        if code_lock2unlock[ind_beg1 + 1 : ind_end1] == "":
            code_lock2unlock = (
                code_lock2unlock[: ind_beg1 + 1]
                + "{0}"
                + "["
                + str(ind_array)
                + "]"
                + code_lock2unlock[ind_end1:]
            )
            code_unlock2lock = (
                code_unlock2lock[: ind_beg2 + 1]
                + "{0}"
                + "["
                + str(iter - 1)
                + "]"
                + code_unlock2lock[ind_end2:]
            )
            ind_array += 1
        iter += 1

    code_lock2unlock = "[" + code_lock2unlock[1:]
    code_unlock2lock = "[" + code_unlock2lock[1:]
    if unlocked_cov is None:
        return (
            np.array(locked_init_params),
            locked_quentities,
            code_lock2unlock,
            code_unlock2lock,
        )
    else:
        return (
            np.array(locked_init_params),
            locked_quentities,
            np.array(locked_cov),
            code_lock2unlock,
            code_unlock2lock,
        )


def gen_unlocked_params(
    locked_init_params, locked_quentities, lock_protocols, locked_cov=None
):
    assert not (
        len(locked_init_params) == 0
        or len(locked_quentities) == 0
        or len(locked_init_params) != len(locked_quentities) == 0
    )
    if not locked_cov is None:
        locked_cov = list(locked_cov)
    locked_init_params = list(locked_init_params)
    unlocked_init_params = locked_init_params.copy()
    unlocked_quentities = locked_quentities.copy()
    if not locked_cov is None:
        unlocked_cov = locked_cov.copy()

    for target, lines in lock_protocols.lock.items():
        for line in lines:
            ind_target = find_nth_occurrence(unlocked_quentities, "I", target + 1) + 1
            ind_line = find_nth_occurrence(unlocked_quentities, "I", line[0] + 1) + 1
            # print(ind_target,locked_init_params[ind_target])
            unlocked_quentities.insert(ind_line, "x")
            unlocked_init_params.insert(
                ind_line, unlocked_init_params[ind_target] + line[1]
            )
            if not locked_cov is None:
                unlocked_cov.insert(ind_line, unlocked_cov[ind_target])

    if locked_cov is None:
        return (np.array(unlocked_init_params), unlocked_quentities)
    else:
        return (np.array(unlocked_init_params), unlocked_quentities, unlocked_cov)


def gen_lock_fit_func(
    unlocked_init_params, unlocked_quentities, lock_protocols, inner_fit_func
):
    _, _, code_lock2unlock, code_unlock2lock = gen_locked_params(
        unlocked_init_params, unlocked_quentities, lock_protocols
    )

    time_str = datetime.datetime.now().strftime("%H%M%d%H%M%S")
    str_func = """
    \nimport numpy as np
    \nfrom SAFFRON.fit_models import flat_inArg_multiGauss
    \ndef func_{}(x,*array):
    \n    unlocked_params = {}
    \n    y = {}(x,*unlocked_params)
    \n    #locked_params = np.array({})
    \n    return y
    """.format(
        time_str,
        code_lock2unlock.format("array"),
        inner_fit_func.__name__,
        code_unlock2lock.format("y"),
    )
    # print(str_func)
    if lock_protocols._dir_tmp_functions is None:
        lock_protocols._dir_tmp_functions = Path("./tmp_functions").resolve()
        lock_protocols._dir_tmp_functions.mkdir(parents=True, exist_ok=True)
        sys.path.append(lock_protocols._dir_tmp_functions)
        sys.path.append(Path("./SAFFRON").resolve())

    str_import_Template = "from tmp_functions.{0} import {0}"
    str_func = (
        "\n".join(
            [
                str_import_Template.format(i.__name__)
                for i in lock_protocols.import_function_list
                if i != flat_inArg_multiGauss
            ]
        )
        + str_func
    )

    # saving to file
    _func_name = "func_" + time_str
    with open(lock_protocols._dir_tmp_functions / (_func_name + ".py"), mode="w") as f:
        f.writelines(str_func)

    loc = {}
    exec(str_import_Template.format(_func_name), globals(), loc)
    # exec(str_func,globals(),loc)
    fit_func = list(loc.items())[0][1]
    return fit_func, str_func
