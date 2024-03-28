#!/bin/bash
PROTO_STATUS=$(protolock status --strict)
if [ -z  "$PROTO_STATUS" ]
then
    VERSION_CHANGE=patch
    echo "Backward compatibility not broken"
else
    VERSION_CHANGE=minor
    echo -e "Backward compatibility is broken!!!\n$(PROTO_STATUS)"
fi

python3 version.py --version-change=$VERSION_CHANGE
protolock commit --force 

cat VERSION