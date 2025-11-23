pipeline {
    // We just run on the default Windows node
    agent any

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out source code from Git...'
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                echo 'Setting up Python environment...'
                // Windows commands -> use bat
                bat 'python --version'
                bat 'pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline to pipeline.yaml...'
                bat 'python pipeline.py'
                // fail build if pipeline.yaml wasn't created
                bat 'if not exist pipeline.yaml ( exit /b 1 )'
            }
        }
    }

    post {
        success {
            echo ' Jenkins CI pipeline finished successfully!'
            archiveArtifacts artifacts: 'pipeline.yaml', fingerprint: true
        }
        failure {
            echo ' Jenkins CI pipeline failed. Check the log above.'
        }
    }
}
