pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'dockerhub'  // This is the ID of your Docker Hub credentials in Jenkins
    }

    stages {
        stage('Login to DockerHub') {
            steps {
                script {
                    docker.withCredentials([usernamePassword(credentialsId: DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        sh 'docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD'
                    }
                }
            }
        }
    }
}
