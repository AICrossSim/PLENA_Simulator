from setuptools import setup, find_packages

# Get packages and create proper package_dir mapping
tools_packages = find_packages("tools")

# Create package directory mapping for nested packages
package_dir = {}
for pkg in tools_packages:
    # Convert dot notation to path (e.g., "quant.quantizer" -> "tools/quant/quantizer")
    package_dir[pkg] = "tools/" + pkg.replace(".", "/")

aria_packages = find_packages("tools/aria-llama-ops/src")
for pkg in aria_packages:
    # Add aria-llama-ops package mapping
    package_dir[pkg] = "tools/aria-llama-ops/src/" + pkg.replace(".", "/")

setup(
    name='llama_coprocessor',
    version='1.0',  # random
    packages=tools_packages + aria_packages,
    package_dir=package_dir,
    install_requires=[
    ]
)