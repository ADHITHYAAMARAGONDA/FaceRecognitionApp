pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'dockerhub'
        SKIP_DEPLOYMENT = 'true' // Set to 'false' when you want to include deployment
    }

    stages {
        stage('Login to DockerHub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: "${DOCKER_CREDENTIALS_ID}", usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh 'docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD'
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t adhithya143/ai-face-recognition:latest .'
            }
        }

        stage('Push to DockerHub') {
            when {
                expression { return env.SKIP_DEPLOYMENT != 'true' }
            }
            steps {
                sh 'docker push adhithya143/ai-face-recognition:latest'
            }
        }
    }
}
