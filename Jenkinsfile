pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'dockerhub' // Jenkins Credentials ID
        IMAGE_NAME = 'adhithya143/ai-face-recognition'
        IMAGE_TAG = 'latest'
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
                script {
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
                }
            }
        }

        stage('Push to DockerHub') {
            steps {
                script {
                    sh "docker push ${IMAGE_NAME}:${IMAGE_TAG}"
                }
            }
        }
    }
}
