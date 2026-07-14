const DEFAULT_ORIGINS=['https://jacktools.online','https://www.jacktools.online'];

function allowedOrigins(env){
  return String(env.ALLOWED_ORIGIN||DEFAULT_ORIGINS.join(','))
    .split(',').map(value=>value.trim()).filter(Boolean);
}

function corsHeaders(origin,env){
  const allowed=allowedOrigins(env);
  const headers={
    'Access-Control-Allow-Methods':'POST, OPTIONS',
    'Access-Control-Allow-Headers':'Content-Type',
    'Access-Control-Max-Age':'86400',
    'Vary':'Origin'
  };
  if(allowed.includes(origin))headers['Access-Control-Allow-Origin']=origin;
  return headers;
}

function json(data,status,origin,env){
  return new Response(JSON.stringify(data),{
    status,
    headers:{'Content-Type':'application/json; charset=utf-8',...corsHeaders(origin,env)}
  });
}

function outputText(response){
  for(const item of response.output||[]){
    for(const part of item.content||[]){
      if(part.type==='output_text'&&part.text)return part.text;
    }
  }
  return '';
}

export default {
  async fetch(request,env){
    const origin=request.headers.get('Origin')||'';
    const path=new URL(request.url).pathname;
    if(request.method==='OPTIONS'){
      if(!allowedOrigins(env).includes(origin))return json({error:'Origin not allowed.'},403,origin,env);
      return new Response(null,{status:204,headers:corsHeaders(origin,env)});
    }
    if(request.method!=='POST'||!['/','/unwrap'].includes(path))return json({error:'Not found.'},404,origin,env);
    if(!allowedOrigins(env).includes(origin))return json({error:'Origin not allowed.'},403,origin,env);
    if(!env.OPENAI_API_KEY)return json({error:'OPENAI_API_KEY has not been configured.'},503,origin,env);
    const length=Number(request.headers.get('Content-Length')||0);
    if(length>20000)return json({error:'Request is too large.'},413,origin,env);

    let payload;
    try{payload=await request.json();}
    catch{return json({error:'Invalid JSON request.'},400,origin,env);}
    if(JSON.stringify(payload).length>20000)return json({error:'Request is too large.'},413,origin,env);
    if(!payload?.mesh||!payload?.preferences)return json({error:'Mesh profile and preferences are required.'},400,origin,env);

    const schema={
      type:'object',
      additionalProperties:false,
      properties:{
        strategy:{type:'string',enum:['balanced','detail','compact']},
        angleThreshold:{type:'integer',minimum:1,maximum:180},
        padding:{type:'integer',minimum:1,maximum:64},
        textureSize:{type:'integer',enum:[1024,2048,4096,8192]},
        rotateIslands:{type:'boolean'},
        summary:{type:'string'}
      },
      required:['strategy','angleThreshold','padding','textureSize','rotateIslands','summary']
    };

    try{
      const openAIResponse=await fetch('https://api.openai.com/v1/responses',{
        method:'POST',
        headers:{
          'Authorization':`Bearer ${env.OPENAI_API_KEY}`,
          'Content-Type':'application/json'
        },
        body:JSON.stringify({
          model:env.OPENAI_MODEL||'gpt-5.4-mini',
          store:false,
          reasoning:{effort:'low'},
          max_output_tokens:600,
          instructions:'You are a specialist 3D UV-unwrapping planner. Review the compact mesh statistics and artist preferences. Return a practical plan for a non-overlapping packed triangle atlas. Favor enough padding for the requested texture resolution, preserve detail on complex meshes, and keep the summary under 35 words. Do not claim to have inspected geometry that was not provided.',
          input:JSON.stringify(payload),
          text:{format:{type:'json_schema',name:'uv_unwrap_plan',strict:true,schema}}
        })
      });
      const result=await openAIResponse.json();
      if(!openAIResponse.ok){
        const message=result?.error?.message||'OpenAI rejected the request.';
        return json({error:message},openAIResponse.status,origin,env);
      }
      const text=outputText(result);
      if(!text)return json({error:'OpenAI returned no unwrap plan.'},502,origin,env);
      const plan=JSON.parse(text);
      return json({plan,usage:result.usage||null},200,origin,env);
    }catch(error){
      console.error('AI unwrap worker error',error);
      return json({error:'The AI unwrap service failed.'},500,origin,env);
    }
  }
};
