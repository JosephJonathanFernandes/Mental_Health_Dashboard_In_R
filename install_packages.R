# Installation script for Mental Health Dashboard
# Run this in VS Code or RStudio

cat("Installing required R packages for Mental Health Dashboard...\n\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Required packages
required_packages <- c(
  "shiny",        # Web framework
  "tidyverse",    # Data manipulation (includes ggplot2, dplyr, tidyr, etc.)
  "caret",        # Machine learning
  "randomForest", # Random Forest algorithm
  "plotly",       # Interactive plots
  "DT",           # Data tables
  "pROC",         # ROC curves
  "recipes",      # Preprocessing pipelines
  "rlang",        # Tidy programming helpers
  "bslib"         # Bootstrap theming
)

# Optional packages for enhanced features
optional_packages <- c(
  "GGally",       # Enhanced pair plots
  "skimr"         # Data summaries
)

cat("========================================\n")
cat("STEP 1: Installing required packages\n")
cat("========================================\n\n")

# Install required packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

cat("\n========================================\n")
cat("STEP 2: Installing optional packages\n")
cat("========================================\n\n")

# Install optional packages
for (pkg in optional_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    tryCatch({
      install.packages(pkg, dependencies = TRUE)
    }, error = function(e) {
      cat(sprintf("Warning: Could not install %s (optional package)\n", pkg))
    })
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

cat("\n========================================\n")
cat("Installation Summary\n")
cat("========================================\n\n")

# Check which packages are installed
installed <- sapply(c(required_packages, optional_packages), function(pkg) {
  requireNamespace(pkg, quietly = TRUE)
})

cat("Required packages:\n")
for (pkg in required_packages) {
  status <- if (installed[pkg]) "✓ Installed" else "✗ Missing"
  cat(sprintf("  %s: %s\n", pkg, status))
}

cat("\nOptional packages:\n")
for (pkg in optional_packages) {
  status <- if (installed[pkg]) "✓ Installed" else "✗ Not installed"
  cat(sprintf("  %s: %s\n", pkg, status))
}

cat("\n========================================\n")

all_required_installed <- all(installed[required_packages])

if (all_required_installed) {
  cat("\n✓ SUCCESS! All required packages are installed.\n")
  cat("You can now run the app with:\n")
  cat("  shiny::runApp('app.R')\n\n")
} else {
  cat("\n✗ Some required packages failed to install.\n")
  cat("Please install them manually or check for errors above.\n\n")
}
