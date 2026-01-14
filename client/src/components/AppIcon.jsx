import React from 'react'
import * as Icons from 'lucide-react'

const Icon = ({ name, className, size = 20, ...props }) => {
  const LucideIcon = Icons[name]
  
  if (!LucideIcon) {
    console.warn(`Icon "${name}" not found in lucide-react`)
    return <Icons.HelpCircle className={className} size={size} {...props} />
  }
  
  return <LucideIcon className={className} size={size} {...props} />
}

export default Icon