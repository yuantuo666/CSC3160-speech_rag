#!/bin/bash

# Function to display menu
show_menu() {
    clear
    echo "=========================================="
    echo "           Speech RAG Manager             "
    echo "=========================================="
    echo "1. Configure Environment (Install Deps)"
    echo "2. Start Gradio App"
    echo "3. Build Test Dataset"
    echo "4. Run Evaluation"
    echo "5. Start Remote vLLM (Optional Helper)"
    echo "6. Exit"
    echo "=========================================="
    echo -n "Please select an option [1-6]: "
}

# Main loop
while true; do
    show_menu
    read choice
    case $choice in
        1)
            echo "Installing dependencies..."
            pip install -r requirements.txt
            echo "Done."
            ;;
        2)
            echo "Starting Gradio App..."
            python app.py
            ;;
        3)
            echo "Building Test Dataset..."
            python -m eval.build_test_dataset
            ;;
        4)
            echo "Running Evaluation..."
            python -m eval.evaluate
            ;;
        5)
            echo "Starting Remote vLLM..."
            # Check if executable
            if [ ! -x "./scripts/start_remote_vllm.sh" ]; then
                chmod +x ./scripts/start_remote_vllm.sh
            fi
            ./scripts/start_remote_vllm.sh
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
    echo ""
    echo "Press Enter to continue..."
    read
done
