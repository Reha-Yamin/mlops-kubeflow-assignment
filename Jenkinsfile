pipeline {
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
                echo 'Installing Python dependencies (user mode)...'
                bat 'python --version'
                bat 'pip --version'

                // Install requirements for user only (NOT system-wide)
                bat 'pip install --user -r requirements.txt'
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline...'
                
                // Use user python path
                bat 'python pipeline.py'

                // Verify pipeline.yaml was created
                bat 'if not exist pipeline.yaml ( exit /b 1 )'
            }
        }
    }

    post {
        success {
            echo "Jenkins CI pipeline finished successfully!"
            archiveArtifacts artifacts: 'pipeline.yaml', fingerprint: true
        }
        failure {
            echo " Jenkins CI pipeline failed. See logs above."
        }
    }
}
