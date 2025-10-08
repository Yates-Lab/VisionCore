#!/bin/bash
# VisionCore Training System - Phase 1 Cleanup Script
# 
# This script removes duplicate files and updates references.
# Run this after reviewing TRAINING_SYSTEM_SUMMARY.md
#
# Usage: bash cleanup_duplicates.sh [--dry-run]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No files will be modified${NC}"
fi

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to safely remove file
safe_remove() {
    local file=$1
    if [[ -f "$file" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "  Would remove: $file"
        else
            echo "  Removing: $file"
            rm "$file"
        fi
    else
        print_warning "File not found: $file"
    fi
}

# Function to check if file exists
check_file() {
    local file=$1
    if [[ ! -f "$file" ]]; then
        print_error "Required file not found: $file"
        exit 1
    fi
}

# ============================================================================
# Pre-flight checks
# ============================================================================

print_step "Running pre-flight checks..."

# Check we're in the right directory
if [[ ! -d "training" ]] || [[ ! -d "models" ]]; then
    print_error "This script must be run from the VisionCore root directory"
    exit 1
fi

# Check that canonical files exist
check_file "training/train_ddp_multidataset.py"
check_file "experiments/run_all_models.sh"

echo -e "${GREEN}✓${NC} Pre-flight checks passed"
echo ""

# ============================================================================
# Step 1: Remove duplicate training script
# ============================================================================

print_step "Step 1: Removing duplicate training script"

DUPLICATE_SCRIPT="jake/multidataset_ddp/train_ddp_multidataset.py"

if [[ -f "$DUPLICATE_SCRIPT" ]]; then
    # Check if it's actually a duplicate
    if diff -q "training/train_ddp_multidataset.py" "$DUPLICATE_SCRIPT" > /dev/null 2>&1; then
        echo "  Files are identical - safe to remove"
        safe_remove "$DUPLICATE_SCRIPT"
    else
        print_warning "Files differ! Manual review needed:"
        echo "  diff training/train_ddp_multidataset.py $DUPLICATE_SCRIPT"
        if [[ "$DRY_RUN" == false ]]; then
            echo "  Skipping removal for safety"
        fi
    fi
else
    echo "  Duplicate script not found (already removed?)"
fi

echo ""

# ============================================================================
# Step 2: Remove duplicate shell scripts
# ============================================================================

print_step "Step 2: Removing duplicate shell scripts"

DUPLICATE_SCRIPTS=(
    "jake/multidataset_ddp/run_all_models.sh"
    "jake/multidataset_ddp/run_all_models_backimage.sh"
    "jake/multidataset_ddp/run_all_models_cones.sh"
    "jake/multidataset_ddp/run_all_models_gaborium.sh"
)

for script in "${DUPLICATE_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        # Extract filename
        filename=$(basename "$script")
        canonical="experiments/$filename"
        
        if [[ -f "$canonical" ]]; then
            # Check if they're identical
            if diff -q "$canonical" "$script" > /dev/null 2>&1; then
                echo "  $filename: identical to canonical"
                safe_remove "$script"
            else
                print_warning "$filename: differs from canonical - manual review needed"
                if [[ "$DRY_RUN" == false ]]; then
                    echo "  Skipping removal for safety"
                fi
            fi
        else
            print_warning "Canonical file not found: $canonical"
        fi
    else
        echo "  $script: not found (already removed?)"
    fi
done

echo ""

# ============================================================================
# Step 3: Check for references to removed files
# ============================================================================

print_step "Step 3: Checking for references to removed files"

echo "  Searching for references to jake/multidataset_ddp/train_ddp_multidataset.py..."
if grep -r "jake/multidataset_ddp/train_ddp_multidataset" . --exclude-dir=.git --exclude="*.md" --exclude="cleanup_duplicates.sh" 2>/dev/null; then
    print_warning "Found references to removed file - these need to be updated"
else
    echo "  No references found"
fi

echo ""
echo "  Searching for references to jake/multidataset_ddp/run_all_models..."
if grep -r "jake/multidataset_ddp/run_all_models" . --exclude-dir=.git --exclude="*.md" --exclude="cleanup_duplicates.sh" 2>/dev/null; then
    print_warning "Found references to removed files - these need to be updated"
else
    echo "  No references found"
fi

echo ""

# ============================================================================
# Step 4: Update shell scripts to use canonical path
# ============================================================================

print_step "Step 4: Updating shell scripts to use canonical training script path"

SHELL_SCRIPTS=(
    "experiments/run_all_models.sh"
    "experiments/run_all_models_backimage.sh"
    "experiments/run_all_models_cones.sh"
    "experiments/run_all_models_gaborium.sh"
    "experiments/run_all_models_pretraining.sh"
)

for script in "${SHELL_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        # Check if it uses the old path
        if grep -q "python train_ddp_multidataset.py" "$script"; then
            echo "  Updating: $script"
            if [[ "$DRY_RUN" == true ]]; then
                echo "    Would change: python train_ddp_multidataset.py"
                echo "    To:           python training/train_ddp_multidataset.py"
            else
                # Create backup
                cp "$script" "$script.backup"
                # Update the path
                sed -i 's|python train_ddp_multidataset.py|python training/train_ddp_multidataset.py|g' "$script"
                echo "    ✓ Updated (backup saved as $script.backup)"
            fi
        else
            echo "  $script: already uses correct path"
        fi
    else
        print_warning "Script not found: $script"
    fi
done

echo ""

# ============================================================================
# Step 5: Summary
# ============================================================================

print_step "Cleanup Summary"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo "No files were modified. Review the output above and run without --dry-run to apply changes."
else
    echo ""
    echo -e "${GREEN}CLEANUP COMPLETE${NC}"
    echo ""
    echo "Files removed:"
    echo "  - jake/multidataset_ddp/train_ddp_multidataset.py (if it existed)"
    echo "  - jake/multidataset_ddp/run_all_models*.sh (if they existed)"
    echo ""
    echo "Files updated:"
    echo "  - experiments/run_all_models*.sh (to use training/train_ddp_multidataset.py)"
    echo "  - Backups saved as *.backup"
    echo ""
    echo "Next steps:"
    echo "  1. Test that training still works:"
    echo "     bash experiments/run_all_models.sh"
    echo ""
    echo "  2. If everything works, remove backups:"
    echo "     rm experiments/*.backup"
    echo ""
    echo "  3. Commit changes:"
    echo "     git add -A"
    echo "     git commit -m 'Remove duplicate training scripts and update paths'"
fi

echo ""
echo -e "${GREEN}Done!${NC}"

