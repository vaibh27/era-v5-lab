# "1.0.13" --> "1.1.0" regression. `filetype.guess` stopped working with output from `read(file_path, "rb")`

Upstream issue: https://github.com/h2non/filetype.py/issues/130 (opened by @simon-liebehenschel on 2022-07-21). Text copied unchanged.

## Reproducible code sample
```python
import filetype

photo_path = "image.jpeg"

with open(str(photo_path), "rb") as photo_object:
    result = filetype.guess(photo_object)
```
## "1.0.13" result

All works as expected:
```
<filetype.types.image.Jpeg object at 0x7fbe8e9d4190>
```
## "1.1.0" result

```
/opt/pysetup/.venv/lib/python3.10/site-packages/filetype/filetype.py:28: in guess
    return match(obj) if obj else None
/opt/pysetup/.venv/lib/python3.10/site-packages/filetype/match.py:29: in match
    buf = get_bytes(obj)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
obj = <_io.BufferedReader name='/builds/codeavors/traveltech/backend_common/tests/google/test_photo_paris.jpeg'>
    def get_bytes(obj):
        """
        Infers the input type and reads the first 262 bytes,
        returning a sliced bytearray.
    
        Args:
            obj: path to readable, file, bytes, bytearray or memoryview.
    
        Returns:
            First 262 bytes of the file content as bytearray type.
    
        Raises:
            TypeError: if obj is not a supported type.
        """
        if isinstance(obj, bytearray):
            return signature(obj)
    
        if isinstance(obj, str):
            return get_signature_bytes(obj)
    
        if isinstance(obj, bytes):
            return signature(obj)
```
