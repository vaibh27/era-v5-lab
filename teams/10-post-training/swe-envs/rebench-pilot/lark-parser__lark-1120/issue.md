# Caching does not work with recursive pattern

Upstream issue: https://github.com/lark-parser/lark/issues/1119 (opened by @gruebel on 2022-02-26). Text copied unchanged.

**Describe the bug**

I tried to leverage caching to speed up the loading time of my grammar, but it didn't work, so I started to dig into the code and found that it can't load the pickled file

```shell
Traceback (most recent call last):
  ...
  File ".../lark/lark.py", line 326, in __init__
    raise e
  File ".../lark/lark.py", line 321, in __init__
    cached_parser_data = pickle.load(f)
TypeError: __new__() missing 1 required positional argument: 'name'
```

I could trace it down to an issue with `MAXREPEAT` and it happens, when a recursive pattern is used.

**To Reproduce**

```python
from lark import Lark


def main():
    json_grammar = r"""
    start: element*
    
    element: value
    
    ?value: "null"                      -> null
          | QUOTED_INTERPOLATION        -> string
    
    QUOTED_INTERPOLATION: /'(?R)'/
    
    %import common.NEWLINE
    %import common.WS_INLINE
    
    %ignore NEWLINE
    %ignore WS_INLINE
    """

    json_parser = Lark(json_grammar, start="start", parser="lalr", regex=True, cache=True)

    tree = json_parser.parse(
        """
        null
        """
    )
    print(tree.pretty())


if __name__ == "__main__":
    main()
```
