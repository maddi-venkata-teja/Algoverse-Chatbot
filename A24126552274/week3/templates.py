"""
templates.py

Instruction templates used to generate diverse prompts for the
synthetic DSA dataset.
"""

QUERY_TEMPLATES = {

    "definition": {

        "beginner": [

            "What is a {name}?",
            "Explain {name} in simple words.",
            "Define {name}.",
            "Can you describe {name}?",
            "What does {name} mean?",
            "What exactly is a {name}?",
            "Give a beginner-friendly explanation of {name}.",
            "Introduce the concept of {name}.",
            "How would you explain {name} to someone new to programming?",
            "What is the purpose of a {name}?",
            "Explain the basic idea behind {name}.",
            "What should every beginner know about {name}?",
            "Describe {name} using simple language.",
            "What kind of data structure is {name}?",
            "How can we understand {name} easily?",
            "Give the definition of {name}."

        ],

        "intermediate": [

            "Explain the internal structure of a {name}.",
            "Describe the memory layout of a {name}.",
            "How is {name} implemented internally?",
            "Explain how a {name} stores data.",
            "Discuss the internal organization of a {name}.",
            "Describe the underlying representation of a {name}.",
            "How does memory allocation work in a {name}?",
            "Explain the storage mechanism of a {name}.",
            "How is a {name} represented in memory?",
            "Discuss the implementation details of a {name}.",
            "Explain the architecture of a {name}.",
            "What structural properties define a {name}?",
            "Describe how elements are organized inside a {name}.",
            "Explain the physical layout of a {name}.",
            "What happens internally when a {name} is created?",
            "How does a {name} manage its data?"

        ],

        "advanced": [

            "Describe the theoretical model of a {name}.",
            "Explain the invariants maintained by a {name}.",
            "Formally define a {name}.",
            "Discuss the mathematical foundation of a {name}.",
            "Explain the theoretical guarantees of a {name}.",
            "Describe the formal properties of a {name}.",
            "What invariants are preserved in a {name}?",
            "How is a {name} defined in theoretical computer science?",
            "Explain the formal abstraction behind a {name}.",
            "Discuss the correctness conditions of a {name}.",
            "What principles govern the design of a {name}?",
            "Describe the formal behavior of a {name}.",
            "Explain the conceptual model of a {name}.",
            "What makes a {name} mathematically valid?",
            "Describe the theoretical characteristics of a {name}.",
            "Explain the abstract model of a {name}."

        ]

    },

    "complexity": {

        "beginner": [

            "What is the time complexity of common operations on a {name}?",
            "How fast are the basic operations in a {name}?",
            "Explain the performance of a {name}.",
            "What is the runtime of operations on a {name}?",
            "How efficient is a {name}?",
            "What are the operation costs of a {name}?",
            "Explain the speed of a {name}.",
            "How long do operations take in a {name}?",
            "Discuss the efficiency of a {name}.",
            "What are the basic complexities of a {name}?",
            "Describe the computational cost of a {name}.",
            "What is the performance profile of a {name}?",
            "How efficient are insertions and searches in a {name}?",
            "Explain the execution cost of a {name}.",
            "Discuss runtime analysis of a {name}.",
            "How does a {name} perform?"

        ],

        "intermediate": [

            "Compare average-case and worst-case complexity of a {name}.",
            "Explain all operation complexities of a {name}.",
            "Analyze the performance of a {name}.",
            "Discuss complexity analysis of a {name}.",
            "Describe average and worst-case behavior of a {name}.",
            "Compare different operation costs of a {name}.",
            "Explain performance trade-offs of a {name}.",
            "Discuss computational complexity of a {name}.",
            "Analyze time and space complexity of a {name}.",
            "Evaluate runtime performance of a {name}.",
            "Describe efficiency under different conditions.",
            "How does workload affect the complexity of a {name}?",
            "Explain complexity in practical systems.",
            "Compare runtime for common operations.",
            "Analyze efficiency across all operations.",
            "Discuss algorithmic performance."

        ],

        "advanced": [

            "Derive the amortized complexity of a {name}.",
            "Explain amortized analysis of a {name}.",
            "Discuss edge cases affecting complexity.",
            "Provide a formal complexity analysis.",
            "Explain asymptotic behavior of a {name}.",
            "Analyze amortized versus worst-case performance.",
            "Derive runtime mathematically.",
            "Discuss theoretical complexity guarantees.",
            "Explain complexity proofs.",
            "Analyze scalability of a {name}.",
            "Discuss amortized performance under heavy workloads.",
            "Explain why amortized analysis is important.",
            "Compare asymptotic bounds.",
            "Analyze large-scale performance.",
            "Evaluate complexity with formal reasoning.",
            "Explain advanced runtime characteristics."

        ]

    },
    "use_case": {

        "beginner": [

            "Give a simple real-world use case for {name}.",
            "Where is {name} used in daily life?",
            "Provide a practical example of {name}.",
            "When should someone use a {name}?",
            "What is a beginner example of {name}?",
            "Describe a real-world application of {name}.",
            "How is {name} useful in software?",
            "Give an easy example using {name}.",
            "Where would you apply {name}?",
            "Explain a simple application of {name}.",
            "Why is {name} useful?",
            "Give a scenario where {name} helps.",
            "Describe a practical situation using {name}.",
            "What problem can {name} solve?",
            "Show a beginner-friendly application of {name}.",
            "Explain when to choose {name}."

        ],

        "intermediate": [

            "Where is {name} commonly used in software systems?",
            "Discuss industrial applications of {name}.",
            "Explain why modern software uses {name}.",
            "In which applications is {name} preferred?",
            "Describe systems built around {name}.",
            "How is {name} used in production software?",
            "Discuss common engineering uses of {name}.",
            "Explain enterprise applications of {name}.",
            "How does {name} help software engineers?",
            "Describe typical use cases of {name}.",
            "Discuss practical deployments of {name}.",
            "Where would an engineer choose {name}?",
            "Explain the role of {name} in applications.",
            "How does {name} improve software design?",
            "What industries use {name} frequently?",
            "Describe medium-scale applications of {name}."

        ],

        "advanced": [

            "Analyze why {name} is chosen in high-performance systems.",
            "Explain architectural trade-offs of {name}.",
            "Discuss enterprise-scale uses of {name}.",
            "Why would a systems architect choose {name}?",
            "Analyze production use of {name}.",
            "Evaluate {name} in large-scale software.",
            "Discuss performance-critical applications of {name}.",
            "Explain system-level usage of {name}.",
            "Why is {name} suitable for scalable systems?",
            "Analyze the engineering decisions behind {name}.",
            "Discuss advanced deployment scenarios.",
            "Evaluate architectural choices involving {name}.",
            "Explain performance engineering using {name}.",
            "Analyze software design using {name}.",
            "Describe enterprise architectures using {name}.",
            "Explain advanced real-world implementations of {name}."

        ]

    },

    "tradeoff": {

        "beginner": [

            "What are the advantages and disadvantages of {name}?",
            "List the pros and cons of {name}.",
            "Why should someone use {name}?",
            "When is {name} not a good choice?",
            "Explain strengths and weaknesses of {name}.",
            "Discuss benefits and drawbacks of {name}.",
            "What makes {name} useful?",
            "What limitations does {name} have?",
            "Describe positive and negative aspects of {name}.",
            "Explain why {name} is sometimes avoided.",
            "Discuss the main features of {name}.",
            "Why do programmers like {name}?",
            "What problems exist with {name}?",
            "Describe common disadvantages of {name}.",
            "Evaluate the usefulness of {name}.",
            "Should beginners learn {name}?"

        ],

        "intermediate": [

            "Compare {name} with a similar data structure.",
            "When would you choose {name} over another structure?",
            "Explain differences between {name} and alternatives.",
            "Compare {name} with competing approaches.",
            "Discuss trade-offs of {name}.",
            "Evaluate {name} against similar structures.",
            "When is {name} the better option?",
            "Compare implementation choices involving {name}.",
            "Discuss advantages over alternative structures.",
            "Explain when another structure is preferable.",
            "Analyze design trade-offs.",
            "Compare efficiency with related structures.",
            "Evaluate practical differences.",
            "Discuss engineering considerations.",
            "Analyze performance trade-offs.",
            "Compare common implementation choices."

        ],

        "advanced": [

            "Critically evaluate the design limitations of {name}.",
            "Discuss mitigation strategies for {name}.",
            "Analyze production limitations of {name}.",
            "Explain how engineers overcome weaknesses of {name}.",
            "Evaluate architectural limitations.",
            "Discuss scalability issues of {name}.",
            "Analyze bottlenecks in {name}.",
            "Describe optimization techniques for {name}.",
            "Explain practical engineering compromises.",
            "Discuss limitations in enterprise software.",
            "Analyze production-level trade-offs.",
            "Evaluate optimization strategies.",
            "Explain advanced implementation challenges.",
            "Discuss real-world engineering constraints.",
            "Analyze large-scale deployment limitations.",
            "Critically assess {name} from a systems perspective."

        ]

    }

}