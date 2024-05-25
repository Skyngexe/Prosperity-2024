#!/bin/sh
cp git_hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
echo "Git hooks set up successfully."
