# b58encode_check does not take a str as input

Upstream issue: https://github.com/keis/base58/issues/45 (opened by @hukkin on 2020-01-13). Text copied unchanged.

b58encode_check only takes `bytes` as input. This is inconsistent with b58encode, which can handle a `Union[str, bytes]` as input.
