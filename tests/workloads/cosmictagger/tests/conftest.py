import pytest
import random

def pytest_addoption(parser):
    parser.addoption('--limit', action='store', default=-1, type=int, help='tests limit')


def pytest_collection_modifyitems(session, config, items):
    limit = config.getoption('--limit')
    if limit < 0:
        return
    essential = []
    non_essential = []
    for item in items:
        if 'build_network' not in item.name:
            essential.append(item)
        else:
            non_essential.append(item)
    random.Random(0).shuffle(non_essential)
    if limit < len(essential):
        selected = essential
    else:
        n = limit - len(essential)
        selected = essential + non_essential[:n]
    selected_ids = {id(item) for item in selected}
    deselected = [item for item in items if id(item) not in selected_ids]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
    # if limit >= 0:
    #     print(type(items[0]))
    #     print(items[0])
    #     print(items[0].__dict__)
    #     items[:] = items[:limit]


# @pytest.fixture(scope="session", autouse=True)
# def callattr_ahead_of_alltests(request):
#     print("callattr_ahead_of_alltests called")
#     seen = {None}
#     session = request.node
#     print(seen)
#     for item in session.items:
#         cls = item.getparent(pytest.Class)
#         if cls not in seen:
#             if hasattr(cls.obj, "callme"):
#                 cls.obj.callme()
#             seen.add(cls)
