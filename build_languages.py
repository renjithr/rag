from tree_sitter import Language

Language.build_library(
    'build/my-languages.so',  # Output path
    ['tree-sitter-swift']     # Path to the Swift grammar
)
