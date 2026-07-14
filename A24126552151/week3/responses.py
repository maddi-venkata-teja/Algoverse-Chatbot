"""
responses.py

Response templates for generating diverse answers.
"""

RESPONSE_TEMPLATES = {

    "definition": [

        (
            "A {name} is a {category} data structure. "
            "{definition} "
            "Think of it like {analogy}. "
            "Basic operations include {basic_ops}."
        ),

        (
            "{name} is defined as {definition}. "
            "It belongs to the {category} category of data structures."
        ),

        (
            "In simple terms, {analogy}. "
            "This is how a {name} works because {definition}."
        ),

        (
            "The {name} stores data {internal_layout}. "
            "Memory is allocated {memory_note}."
        ),

        (
            "{definition} "
            "Its important characteristics are {struct_props}."
        ),

        (
            "From a theoretical perspective, "
            "{theoretical_note}. "
            "Its invariants include {invariants}."
        ),

        (
            "Internally, a {name} stores data {internal_layout}. "
            "This allows {struct_props}."
        ),

        (
            "A {name} can be understood as {definition}. "
            "It is commonly introduced using the analogy of {analogy}."
        )

    ],

    "complexity": [

        (
            "Basic operation complexities are:\n"
            "{ops_basic}\n"
            "Space complexity: {space_complexity}."
        ),

        (
            "Complete complexity profile:\n"
            "{ops_full}\n"
            "Space: {space_complexity}."
        ),

        (
            "{complexity_note}\n"
            "Space required is {space_complexity}."
        ),

        (
            "The amortized analysis is:\n"
            "{ops_advanced}\n"
            "{amortized_note}"
        ),

        (
            "Performance summary:\n"
            "{ops_full}"
        ),

        (
            "The structure has space complexity "
            "{space_complexity}. "
            "{complexity_note}"
        ),

        (
            "Time complexity depends on the operation being performed.\n"
            "{ops_full}"
        ),

        (
            "For advanced analysis:\n"
            "{ops_advanced}"
        )

    ],

    "use_case": [

        (
            "{beginner_use_case}. "
            "This works because {beginner_reason}."
        ),

        (
            "Applications include {applications}. "
            "{intermediate_reason}."
        ),

        (
            "Real-world example: {real_world}."
        ),

        (
            "{advanced_scenario}. "
            "{advanced_reason}."
        ),

        (
            "A practical use case is {beginner_use_case}."
        ),

        (
            "{applications}. "
            "These systems prefer {name} because {intermediate_reason}."
        ),

        (
            "Production systems often use {name}. "
            "{advanced_reason}"
        ),

        (
            "{real_world}. "
            "This demonstrates why {name} is valuable."
        )

    ],

    "tradeoff": [

        (
            "Advantages:\n"
            "{advantages}\n\n"
            "Disadvantages:\n"
            "{disadvantages}"
        ),

        (
            "Compared with {alt_structure}:\n"
            "Prefer {name} when {prefer_this}.\n"
            "Prefer {alt_structure} when {prefer_alt}."
        ),

        (
            "{key_diff}."
        ),

        (
            "Limitations include {limitations}. "
            "These are mitigated by {mitigations}."
        ),

        (
            "{advantages}\n"
            "{disadvantages}"
        ),

        (
            "Production systems solve these issues by "
            "{mitigations}."
        ),

        (
            "Example from industry:\n"
            "{prod_example}"
        ),

        (
            "Choosing between {name} and {alt_structure} "
            "depends on workload requirements."
        )

    ]

}