#!/bin/bash

echo "Script to compile llama.cpp with gcc 8.5 on the Jetson Nano"

# List of packages to check
packages=("nano" "curl" "libcurl4-openssl-dev" "python3-pip" "git")

# Arrays to store results
installed_packages=()
missing_packages=()

echo "Checking installed packages..."

# Loop through each package and check if it is installed
for package in "${packages[@]}"
do
    if dpkg -l | grep -qw "$package"; then
        # Get the version number of the installed package
        version=$(dpkg -l | grep "$package" | awk '{print $3}')
        installed_packages+=("$package ($version)")
    else
        missing_packages+=("$package")
    fi
done

# Display installed packages with versions
if [ ${#installed_packages[@]} -gt 0 ]; then
    echo "Installed packages:"
    for pkg in "${installed_packages[@]}"
    do
        echo " - $pkg"
    done
else
    echo "No packages are installed."
fi

# Display packages that need to be installed
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "Packages that need to be installed:"
    for pkg in "${missing_packages[@]}"
    do
        echo " - $pkg"
    done
else
    echo "All packages are installed."
fi
