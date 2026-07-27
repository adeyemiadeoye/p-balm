from setuptools import setup, find_packages

setup(
    name="pbalm",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description="Software package accompanying PBALM paper (with examples).",
    author="Adeyemi D. Adeoye, Puya Latafat, Alberto Bemporad",
    maintainers = "Adeyemi D. Adeoye, Puya Latafat, Alberto Bemporad",
    keywords = ["optimization", "nonlinear optimization", "constrained optimization", "augmented Lagrangian"],
    author_email="adeyemi.adeoye@imtlucca.it, puya.latafat@imtlucca.it, alberto.bemporad@imtlucca.it",
    url="https://github.com/adeyemiadeoye/p-balm",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "alpaqa==1.1.0a1" # for an efficient PANOC implementation and associated regularizers
    ],
    python_requires=">=3.8",
    license="Apache-2.0",
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    project_urls = {
        "Documentation": "https://adeyemiadeoye.github.io/p-balm/",
        "Source": "https://github.com/adeyemiadeoye/p-balm",
        "Issue Tracker": "https://github.com/adeyemiadeoye/p-balm/issues"
    },

)