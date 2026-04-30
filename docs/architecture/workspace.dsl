workspace "Anemia Prediction System" "C4 architecture model for a CNN-based childhood anemia risk prediction system." {

    model {
        user = person "User" "Uses the mobile application to submit fingernail images and review anemia risk predictions."

        anemiaSystem = softwareSystem "Anemia Prediction System" "Mobile and backend system for early childhood anemia risk prediction from fingernail images. It supports prediction, not clinical diagnosis." {
            
            mobileApp = container "Mobile App" "Captures or selects fingernail images, sends prediction requests, and displays results/history." "React Native / Expo"

            backend = container "Backend API" "Provides prediction, authentication/profile, health, model evaluation, and metrics endpoints." "FastAPI + Uvicorn + Docker" {
                apiRoutes = component "API Routes" "HTTP endpoints for authentication/profile, prediction (POST + history + signed image URL), health, model evaluation, and metrics." "FastAPI routers + lifespan in backend/api/app.py"
                rateLimitMiddleware = component "Rate Limit Middleware" "In-memory sliding-window throttle for sensitive POST routes (/auth/login, /auth/register, /predict)." "Starlette BaseHTTPMiddleware"
                authService = component "Auth Service" "Handles registration, login, current-user lookup, and authenticated Supabase sessions." "Python service"
                profileService = component "Profile Service" "Manages profile data for authenticated users." "Python service"
                authRepository = component "Auth Repository" "Wraps Supabase Auth operations for sign-up, sign-in, and user resolution." "Supabase Auth client"
                profileRepository = component "Profile Repository" "Reads and updates profile rows." "Supabase PostgREST client"
                predictionService = component "Prediction Service" "Orchestrates image validation, CNN inference, calibration, risk mapping, storage, and persistence (POST /predict, history, signed URL)." "Python service"
                imageValidation = component "Image Validation" "Validates MIME/size limits, decodes and resizes the image, and runs a fingernail-presence heuristic before inference." "Python / Pillow / TensorFlow utilities"
                inferenceRuntime = component "Inference Runtime" "Owns model lifecycle (init/shutdown), resolves the active image predictor, and reports model_loaded / degraded status to health and metrics." "Python module + FastAPI lifespan"
                cnnPredictor = component "CNN Predictor" "ImagePredictor protocol with KerasImagePredictor as the production implementation; runs the shared G9 preprocessing pipeline and produces a raw sigmoid probability." "TensorFlow / Keras / MobileNetV2 + ml.preprocessing"
                calibration = component "Probability Calibration" "Applies temperature scaling and operational thresholding." "Python"
                riskMapping = component "Risk Mapping" "Maps calibrated probability into prediction and risk labels." "Python"
                predictionRepository = component "Prediction Repository" "Persists and retrieves prediction records." "Supabase PostgREST client"
                storageRepository = component "Prediction Image Storage" "Uploads prediction images and issues signed download URLs." "Supabase Storage client"
                modelEvaluation = component "Model Evaluation Service" "Exposes static evaluation metrics and calibration metadata for model v1.0." "Python"
                metricsComponent = component "Metrics / Health / Logging" "Prometheus HTTP middleware, /metrics builder, /health status, and structured logs for local and deployed monitoring." "Prometheus client / Starlette middleware / Python logging"
            }

            mlPipeline = container "ML Pipeline" "Offline scripts for dataset preparation, model training, evaluation, calibration, artifact generation, shared preprocessing (G9), TFLite offline inference (G8), and Grad-CAM (G10)." "TensorFlow / Keras / optional MLflow / Python scripts"

            observability = container "Observability Stack" "Collects and visualizes backend metrics for local validation and thesis/demo monitoring." "Prometheus + Grafana"

            tfliteArtifact = container "TFLite Mobile Artifact" "Exported float32 TFLite graph (raw sigmoid only). Temperature scaling and operational threshold from metadata JSON are applied off-device, matching backend calibration. Bundled for optional offline inference." "TensorFlow Lite file + metadata JSON"
        }

        supabase = softwareSystem "Supabase" "Managed backend platform providing Auth, PostgreSQL, Storage, and Row-Level Security." "External" {
            supabaseAuth = container "Supabase Auth" "Authenticates application users." "Supabase Auth"
            supabaseDb = container "PostgreSQL Database" "Stores profiles and prediction records." "Supabase PostgreSQL"
            supabaseStorage = container "Object Storage" "Stores uploaded fingernail images." "Supabase Storage"
        }

        user -> mobileApp "Uses"

        mobileApp -> backend "Submits image and receives calibrated prediction" "HTTPS / multipart-form-data / JSON"
        backend -> supabaseAuth "Validates authenticated user" "JWT / Supabase client"
        backend -> supabaseDb "Stores and retrieves prediction/profile data" "PostgREST"
        backend -> supabaseStorage "Uploads prediction images" "Supabase Storage API"

        observability -> backend "Scrapes /metrics" "Prometheus scrape"

        mlPipeline -> backend "Produces final Keras model and evaluation metadata" ".keras / JSON / Markdown"
        mlPipeline -> tfliteArtifact "Exports mobile inference artifact" ".tflite + metadata JSON"

        mobileApp -> tfliteArtifact "Can use for offline inference mode" "On-device inference"

        mobileApp -> rateLimitMiddleware "Calls backend HTTP endpoints" "HTTPS / JSON / multipart-form-data"
        rateLimitMiddleware -> apiRoutes "Forwards allowed requests"
        apiRoutes -> predictionService "Delegates prediction requests"
        apiRoutes -> authService "Delegates authentication requests"
        apiRoutes -> profileService "Delegates profile requests"
        apiRoutes -> modelEvaluation "Returns model evaluation metadata"
        apiRoutes -> metricsComponent "Exposes /health and /metrics"

        authService -> authRepository "Delegates Supabase Auth operations"
        authService -> profileService "Checks profile completion after auth flows"
        authService -> profileRepository "Ensures profile row on registration"
        profileService -> profileRepository "Reads and updates profile data"
        authRepository -> supabaseAuth "Registers, signs in, and resolves users"
        profileRepository -> supabaseDb "Reads and updates profile rows"
        predictionService -> imageValidation "Validates and prepares image"
        predictionService -> inferenceRuntime "Resolves the active image predictor"
        predictionService -> cnnPredictor "Gets raw sigmoid probability"
        predictionService -> calibration "Applies temperature scaling"
        predictionService -> riskMapping "Maps calibrated probability to risk"
        predictionService -> predictionRepository "Persists prediction record"
        predictionService -> storageRepository "Uploads image and issues signed URLs"

        inferenceRuntime -> cnnPredictor "Loads and owns the Keras predictor instance"
        mlPipeline -> inferenceRuntime "Provides final Keras model artifact" ".keras"
        mlPipeline -> cnnPredictor "Shares preprocessing pipeline (G9)" "ml.preprocessing module"
        predictionRepository -> supabaseDb "Inserts/selects prediction rows"
        storageRepository -> supabaseStorage "Uploads image objects"
        metricsComponent -> inferenceRuntime "Reads model_loaded / service status"
    }

    views {
        theme https://static.structurizr.com/themes/default/theme.json

        systemContext anemiaSystem "C1-System-Context" {
            include *
            autoLayout lr
            title "C1 - System Context: Anemia Prediction System"
            description "System context diagram showing the user, anemia prediction system, and external managed platform."
        }

        container anemiaSystem "C2-Containers" {
            include *
            autoLayout lr
            title "C2 - Container Diagram: Anemia Prediction System"
            description "Container diagram showing the mobile app, backend API, offline ML pipeline, observability stack, exported TFLite artifact, and Supabase services."
        }

        component backend "C3-Backend-Components" {
            include *
            autoLayout lr
            title "C3 - Component Diagram: Backend API"
            description "Component diagram showing the main FastAPI backend modules involved in prediction, authentication/profile, repositories, inference runtime, calibration, persistence, rate limiting, metrics, and model evaluation."
        }

        styles {
            element "Person" {
                shape person
                background #08427B
                color #FFFFFF
            }

            element "Software System" {
                background #1168BD
                color #FFFFFF
            }

            element "Container" {
                background #438DD5
                color #FFFFFF
            }

            element "Component" {
                background #85BBF0
                color #000000
            }

            element "External" {
                background #999999
                color #FFFFFF
            }
        }
    }
}