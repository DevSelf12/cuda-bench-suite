#!/bin/bash
# Upload all files to GitHub via API
REPO="DevSelf12/cuda-bench-suite"

upload_file() {
    local filepath="$1"
    local content
    content=$(base64 -w0 "$filepath" 2>/dev/null || base64 -i "$filepath" 2>/dev/null)
    local msg="Add $filepath"
    
    result=$(gh api "repos/$REPO/contents/$filepath" --method PUT \
        -f message="$msg" \
        -f content="$content" 2>&1)
    
    if echo "$result" | grep -q '"sha"'; then
        echo "OK: $filepath"
    else
        echo "FAIL: $filepath - $(echo "$result" | head -1)"
    fi
}

# Upload all files
for f in $(find . -type f -not -path './.git/*' | sort); do
    filepath="${f#./}"
    upload_file "$filepath"
done
