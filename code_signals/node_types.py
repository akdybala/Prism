IF_STATEMENT = "if_statement"
ELIF_CLAUSE = "elif_clause"
ELSE_CLAUSE = "else_clause"
FOR_STATEMENT = "for_statement"
WHILE_STATEMENT = "while_statement"
TRY_STATEMENT = "try_statement"
EXCEPT_CLAUSE = "except_clause"
FINALLY_CLAUSE = "finally_clause"
WITH_STATEMENT = "with_statement"
MATCH_STATEMENT = "match_statement"
CASE_CLAUSE = "case_clause"

NESTING_TYPES = {
    IF_STATEMENT,
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    FOR_STATEMENT,
    WHILE_STATEMENT,
    TRY_STATEMENT,
    EXCEPT_CLAUSE,
    FINALLY_CLAUSE,
    WITH_STATEMENT,
}
LOOP_TYPES = {FOR_STATEMENT, WHILE_STATEMENT}
BRANCH_TYPES = {
    IF_STATEMENT,
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    "conditional_expression",
}

FUNCTION_DEFINITION = "function_definition"
CLASS_DEFINITION = "class_definition"
DECORATED_DEFINITION = "decorated_definition"
DECORATOR = "decorator"
LAMBDA = "lambda"

CALL = "call"
IDENTIFIER = "identifier"
ATTRIBUTE = "attribute"
CONDITIONAL_EXPRESSION = "conditional_expression"
BOOLEAN_OPERATOR = "boolean_operator"
NOT_OPERATOR = "not_operator"

ASSIGNMENT = "assignment"
AUGMENTED_ASSIGNMENT = "augmented_assignment"
COMPREHENSION_TYPES = {
    "list_comprehension",
    "set_comprehension",
    "dictionary_comprehension",
    "generator_expression",
}

RETURN_STATEMENT = "return_statement"
BREAK_STATEMENT = "break_statement"
CONTINUE_STATEMENT = "continue_statement"
AWAIT = "await"

GLOBAL_STATEMENT = "global_statement"
NONLOCAL_STATEMENT = "nonlocal_statement"
IMPORT_STATEMENT = "import_statement"
IMPORT_FROM_STATEMENT = "import_from_statement"
ERROR = "ERROR"
