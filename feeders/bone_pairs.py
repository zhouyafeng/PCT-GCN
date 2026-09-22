
class BonePairs:

    _ntu_default_pairs = (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
        (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
        (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
        (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
        (20, 20), (21, 7), (22, 7), (23, 11), (24, 11),
    )
    _ntu_chain_pairs = (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
        (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
        (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
        (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
        (20, 20), (21, 22), (22, 7), (23, 24), (24, 11),
    )
    _ntu_kbone_pairs = (
        (0, 20), (1, 1), (2, 2), (3, 20), (4, 4),
        (5, 20), (6, 4), (7, 5), (8, 8), (9, 20),
        (10, 8), (11, 9), (12, 1), (13, 0), (14, 12),
        (15, 13), (16, 1), (17, 0), (18, 16), (19, 17),
        (20, 20), (21, 7), (22, 6), (23, 11), (24, 10),
    )
    _ntu = {
        'default': _ntu_default_pairs,
        'chain': _ntu_chain_pairs,
        'kbone': _ntu_kbone_pairs,
    }

    _nw_ucla = {
        'default': (
            (0, 1), (1, 2), (2, 2), (3, 2), (4, 2),
            (5, 4), (6, 5), (7, 6), (8, 2), (9, 8),
            (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
        ),
        'kbone': (
            (0, 2), (1, 1), (2, 2), (3, 3), (4, 4),
            (5, 2), (6, 4), (7, 5), (8, 8), (9, 2),
            (10, 8), (11, 9), (12, 1), (13, 0), (14, 12),
            (15, 13), (16, 1), (17, 0), (18, 16), (19, 17),
        ),
    }

    _coco_new = {
        'default': (
            (0, 19), (1, 0), (2, 0), (3, 1), (4, 2),
            (5, 19), (6, 19), (7, 5), (8, 6), (9, 7),
            (10, 8), (11, 17), (12, 17), (13, 11), (14, 12),
            (15, 13), (16, 14), (17, 18), (18, 19), (19, 19),
        ),
    }

    _registry = {
        'ntu': _ntu,
        'nw_ucla': _nw_ucla,
        'coco_new': _coco_new,
    }

    @classmethod
    def get(cls, dataset: str, mode: str = 'default') -> tuple:
        if dataset not in cls._registry:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available: {list(cls._registry)}"
            )
        modes = cls._registry[dataset]
        if mode not in modes:
            raise ValueError(
                f"Unknown mode '{mode}' for dataset '{dataset}'. "
                f"Available: {list(modes)}"
            )
        return modes[mode]

    @classmethod
    def available_modes(cls, dataset: str) -> list:
        if dataset not in cls._registry:
            raise ValueError(f"Unknown dataset '{dataset}'.")
        return list(cls._registry[dataset])


def get_bone_pairs(dataset: str, mode: str = 'default') -> tuple:
    return BonePairs.get(dataset, mode)
