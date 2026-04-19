Project: FastAPI Backend

Context:
	•	Backend API for anemia risk detection
	•	Built with FastAPI
	•	Uses Supabase for authentication, database, and storage
	•	Architecture: modular monolith

Guidelines:
	•	Do not add new dependencies without approval
	•	Do not place business logic inside routers
	•	Use services layer for business logic
	•	Prioritize clarity over automation
	•	Keep code simple and maintainable
	•	Do not break existing endpoints