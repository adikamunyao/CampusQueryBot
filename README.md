# CampusQueryBot

CampusQueryBot is an intelligent campus chatbot designed to assist with a wide range of queries based on documents and information provided. Leveraging the power of embeddings and OpenAI's GPT-4, CampusQueryBot delivers precise and contextually relevant responses.

## Features

- **Advanced Document Retrieval**: Utilizes embedding-based techniques to efficiently retrieve relevant documents and information.
- **Customizable Greetings**: Personalize interactions with customizable responses for various greetings and initial queries.
- **GPT-4 Integration**: Harnesses the capabilities of OpenAI's GPT-4 for generating accurate and contextually aware answers.

## Installation

To set up CampusQueryBot on your local machine, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/CampusQueryBot.git
    ```
2. **Navigate to the Project Directory**:
    ```bash
    cd CampusQueryBot
    ```
3. **Install Dependencies**:
    Make sure you have Python 3.8 or later installed. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **API Keys**: Obtain your OpenAI API key and add it to the configuration file.
2. **Document Setup**: Place your documents in the designated directory and ensure they are properly indexed.

## Usage

Start the bot with the following command:
```bash
python run_bot.py
```

Interact with the bot through the command line or integrate it into your web application using the provided API.

## Customization

- **Greeting Responses**: Modify the greeting responses in the `config/greetings.json` file to suit your needs.
- **Document Retrieval Settings**: Adjust settings for document retrieval and embeddings in `config/retrieval_config.json`.

## Contributing

We welcome contributions to enhance CampusQueryBot. To contribute:

1. **Fork the Repository**
2. **Create a New Branch**:
    ```bash
    git checkout -b feature/your-feature
    ```
3. **Make Your Changes and Commit**:
    ```bash
    git add .
    git commit -m "Add a new feature"
    ```
4. **Push to Your Fork**:
    ```bash
    git push origin feature/your-feature
    ```
5. **Create a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to [adikamunyao.email@example.com](mailto:adikamunyao@gmail.com).

---
