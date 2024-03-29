from mmengine.structures import BaseDataElement


class MultilevelPixelData(BaseDataElement):
    def __init__(self, *, metainfo: dict = None, **kwargs) -> None:
        super().__init__(metainfo=metainfo, **kwargs)
        object.__setattr__(self, '_nlevel', None)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get(f'_level_{item}')
        elif isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            return self.data[start:stop:step]
        else:
            raise TypeError(f'Got invalid index type {type(item)}')

    def shape(self):
        return tuple(level.shape for level in self.levels())
