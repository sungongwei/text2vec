const  json = require('./a.json')
const fs = require('fs')
const obj = { }

json.forEach(item=>{
  if(!obj[item.answer]){
    obj[item.answer] = {
      question: [],
      answer: item.answer
    }
  }
  obj[item.answer].question.push(item.question)
})


fs.writeFileSync('./class.json', JSON.stringify(Object.values(obj)))