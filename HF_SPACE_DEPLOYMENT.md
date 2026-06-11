# Deploying Ghost Diary to Hugging Face Spaces

Since running a 3 Billion parameter model locally on a laptop is very demanding on your battery and RAM, moving the `api.py` backend to the cloud is a great idea. Hugging Face Spaces provides an excellent environment for this, especially with their Docker templates!

I have already created a `Dockerfile` and updated `api.py` to use `llama-cpp-python` and the official GGUF model. This means it's ready to run completely free and extremely fast on CPU. Here is exactly how to deploy it:

## 1. Create the Hugging Face Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name it something like `ghost-diary-api`.
3. For the **Space SDK**, select **Docker** and choose the **Blank** template.
4. **Space Hardware:** Choose the free tier (`CPU basic - 2 vCPU · 16GB`). Because we are using the extremely efficient 4-bit GGUF model via `llama-cpp-python`, it will run lightning-fast right out of the box on this free CPU! You do not need to upgrade or pay for a GPU.

## 2. Push Your Code
Once the Space is created, Hugging Face will give you a Git URL (e.g., `https://huggingface.co/spaces/YourUsername/ghost-diary-api`). 

Open your terminal and run these commands to push the backend code to the Space:

```bash
# Add the Hugging Face space as a remote destination
git remote add hf https://huggingface.co/spaces/YourUsername/ghost-diary-api

# Push the code!
git push hf master:main
```
*Note: If you get an authentication error, you'll need to generate an Access Token from your Hugging Face account settings and use it as your password.*

Hugging Face will automatically see the `Dockerfile` and build the server. You'll see it say "Building" and then "Running" on the webpage.

## 3. Connect Your Frontend
Once the Space is running, click the three dots (`...`) in the top right of the Hugging Face Space and select **"Embed this Space"**. You will see the direct URL for your API (it usually looks something like `https://yourusername-ghost-diary-api.hf.space`).

Finally, open `frontend/src/App.tsx` and change the fetch URL:

```typescript
// Change this:
const res = await fetch("/api/chat", { ... })

// To this (using your actual HF Space URL):
const res = await fetch("https://yourusername-ghost-diary-api.hf.space/api/chat", { ... })
```

*Note: You can remove the `/api` proxy block from `vite.config.ts` once you do this since Vite no longer needs to proxy local requests!*
