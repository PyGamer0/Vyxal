from abc import ABC, abstractmethod


class IGenerator(ABC):    
    def __iter__(self):
        return self

    @abstractmethod
    def __copy__(self) -> IGenerator:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def ordered_contains(self, elem) -> bool:
        """
        Whether or not this generator contains elem. Treats the generator as
        being monotonic, unlike __contains__.
        """
        pass

    @abstractmethod
    def __contains__(self, elem) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, position):
        pass

    @abstractmethod
    def __str__(self, limit=-1) -> str:
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """Can another item be gotten from this generator?"""
    
    @abstractmethod
    def reset_index(self, ind):
        """Reset the index to the given index"""

    @abstractmethod
    def to_list(self) -> list:
        """Turn into a list"""


class Generator(IGenerator):
    def __init__(
        self,
        raw_generator,
        limit=-1,
        initial=[],
        is_numeric_sequence=False,
    ):
        self.next_index = 0
        self.end_reached = False
        self.is_numeric_sequence = is_numeric_sequence
        self.do_print = True
        if "__name__" in dir(raw_generator) and type(raw_generator) != Python_Generator:
            if raw_generator.__name__.startswith(
                "FN_"
            ) or raw_generator.__name__.startswith("_lambda"):
                # User defined function
                def gen():
                    generated = initial
                    factor = len(initial)
                    for item in initial:
                        yield item
                    while True:
                        if len(generated) >= (limit + factor) and limit > 0:
                            break
                        else:
                            ret = raw_generator(generated[::-1], arity=len(generated))
                            generated.append(ret[-1])
                            yield ret[-1]

                self.gen = gen()
            elif type(raw_generator) is Function:
                self.gen = raw_generator()
            else:

                def gen():
                    index = 0
                    while True:
                        yield raw_generator(index)
                        index += 1

                self.gen = gen()
        else:

            def niceify(item):
                t_item = vy_type(item)
                if t_item not in [Generator, list, Number, str]:
                    return list(item)
                return item

            self.gen = map(niceify, raw_generator)
        self.generated = []

    def __call__(self, *args, **kwargs):
        return self

    def __contains__(self, item):
        if self.is_numeric_sequence:
            if item in self.generated:
                return True
            temp = next(self)
            while temp <= item:
                temp = next(self)
            return item in self.generated
        else:
            for temp in self:
                if item == temp:
                    return True
            return False

    def __getitem__(self, position):
        if type(position) is slice:
            ret = []
            stop = position.stop or self.__len__()
            start = position.start or 0

            if start < 0 or stop < 0:
                self.generated += list(self.gen)
                return self.generated[position]
            if stop < 0:
                stop = self.__len__() - position.stop - 2

            if position.step and position.step < 0:
                start, stop = stop, start
                stop -= 1
                start -= 1
            # print(start, stop, position.step or 1)
            for i in range(start, stop, position.step or 1):
                ret.append(self.__getitem__(i))
                # print(self.__getitem__(i))
            return ret
        if position < 0:
            self.generated += list(self.gen)
            return self.generated[position]
        if position < len(self.generated):
            return self.generated[position]
        while len(self.generated) < position + 1:
            try:
                self.__next__()
            except:
                self.end_reached = True
                position = position % len(self.generated)

        return self.generated[position]

    def __setitem__(self, position, value):
        if position >= len(self.generated):
            self.__getitem__(position)
        self.generated[position] = value

    def __len__(self):
        return len(self._dereference())

    def __next__(self):
        f = next(self.gen)
        self.generated.append(f)
        return f

    def __iter__(self):
        return self

    def _filter(self, function):
        index = 0
        length = self.__len__()
        while index != length:
            obj = self.__getitem__(index)
            ret = safe_apply(function, obj)
            if ret:
                yield obj
            index += 1

    def _reduce(self, function):
        def ensure_singleton(function, left, right):
            ret = safe_apply(function, left, right)
            if type(ret) in [Generator, list]:
                return ret[-1]
            return ret

        return functools.reduce(
            lambda x, y: ensure_singleton(function, x, y), self._dereference()
        )

    def _dereference(self):
        """
        Only call this when it is absolutely neccesary to convert to a list.
        """
        d = self.generated + list(self.gen)
        self.gen = iter(d[::])
        self.generated = []
        return d

    def _print(self, end="\n"):
        from vyxal.builtins import vy_print

        main = self.generated
        try:
            f = next(self)
            # If we're still going, there's stuff in main that needs printing before printing the generator
            vy_print("⟨", end="")
            for i in range(len(main)):
                vy_print(main[i], end="|" * (i >= len(main)))
            while True:
                try:
                    f = next(self)
                    vy_print("|", end="")
                    vy_print(f, end="")
                except:
                    break
            vy_print("⟩", end=end)

        except:
            vy_print(main, end=end)

    def zip_with(self, other):
        return Generator(zip(self.gen, iter(other)))

    def safe(self):
        import copy

        return copy.deepcopy(self)

    def __str__(self):
        return "⟨" + "|".join(str(item for item in self.generated)) + "...⟩"

    def limit_to_items(self, n):
        out = "⟨"
        item_count = 0
        while not self.end_reached and item_count <= n:
            item = self.__getitem__(item_count)
            if self.end_reached:
                break
            out += (
                str(item) if vy_type(item) is not Generator else item.limit_to_items(n)
            )
            item_count += 1
            out += "|"

        if item_count > n:
            out += "..."

        return out + "⟩"

class IterateGenerator(IGenerator):
    """
    A generator made by repeatedly applying a function to a value
    """
    def __init__(fn, start, ended=False):
        self._generated = [start]
        self.index = 0
        self.ended = ended
    
    @abstractmethod
    def __copy__(self) -> IterateGenerator:
        res = IterateGenerator(self.fn, None)
        res._generated = self._generated[::]
        return res

    @abstractmethod
    def __len__(self) -> int:
        while self.has_next():
            self.__next__()
        
        return len(self.generated)

    def ordered_contains(self, elem) -> bool:
        pass

    @abstractmethod
    def __contains__(self, elem) -> bool:
        if elem in self.generated:
            return True
        
        while self.has_next():
            if self.__next__() == elem:
                return True
        return False

    @abstractmethod
    def __getitem__(self, position):
        pass

    @abstractmethod
    def __str__(self, limit=-1) -> str:
        pass

    @abstractmethod
    def __next__(self):
        self.ind += 1
        if self.ind < len(self._generated):
            return self._generated[self.ind]
        
        next_elem = self.fn(self._generated[-1])
        self._generated.append(next_elem)
        return next_elem

    @abstractmethod
    def __setitem__(self, index, value):
        pass

    def reset_index(self, ind):
        self.ind = ind

    @abstractmethod
    def to_list(self) -> list:
        pass
